"""깊이 로그 보정본 만들기 (원본은 읽기만 하고, 새 폴더에 같은 형식으로 저장).

프레임마다 순서대로:
  1) 경계 튐 점(flying pixel) 제거 — 앞뒤 물체 사이에 생기는 가짜 줄무늬 점
  2) 작은 구멍 채우기 — 넓이 HOLE_MAX_PX 이하의 무효 영역만 (하늘·측정범위 밖 같은 큰 영역은 그대로)
  3) 경계 보존 평활화 — log(깊이)에 bilateral 필터 (깊이 차가 큰 경계는 섞지 않음)
  4) 시간 필터 — 직전 프레임과 3% 이내로 같으면 섞어서 떨림 감소, 다르면(움직임) 새 값 그대로
출력: <원본폴더>_enhanced/ 에 depth_*.npy(uint16, mm) + meta.json(+enhance 설정) + index.csv(통계 재계산)

사용 예:
  python depth3d/enhance.py --start 9000 --end 10500     # 일부 구간
  python depth3d/enhance.py                              # 전체 (원본과 같은 약 32GB)
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_LOG, DepthLog

BASELINE_M = 0.050          # D435 스테레오 기준선

# 스테레오 오차는 거리²에 비례하지만 시차(disparity, 픽셀) 오차는 거리와 무관하게 거의 일정 →
# 모든 필터를 시차 영역에서 고정 임계값으로 수행 (RealSense 공식 후처리와 같은 방식)
P = dict(
    z_min=0.2, z_max=15.0,
    edge_disp_px=1.5,       # 3x3 안 시차 범위가 이보다 크면 경계 튐 점으로 제거
    hole_max_px=300,        # 이 넓이 이하 구멍만 채움
    hole_sigma_px=3.0,
    bilateral_d=9, bilateral_sigma_disp=0.25, bilateral_sigma_px=4.0,
    temporal_sigma_disp=0.25, temporal_alpha=0.4,   # 직전 프레임 가중(최대)
)


def remove_flying(d, valid):
    big = np.where(valid, d, 0).astype(np.float32)
    small = np.where(valid, d, 1e6).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    rng = cv2.dilate(big, k) - cv2.erode(small, k)
    return valid & ~(rng > P["edge_disp_px"])


def fill_small_holes(d, valid):
    holes = (~valid).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= P["hole_max_px"]
    fill = small[lab] & (holes > 0)
    if not fill.any():
        return d, valid
    # 정규화 컨볼루션: 주변 유효 시차의 가우시안 가중 평균 (큰 구멍은 대상 아님)
    s = P["hole_sigma_px"]
    num = cv2.GaussianBlur(np.where(valid, d, 0).astype(np.float32), (0, 0), s)
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), s)
    ok = fill & (den > 0.05)
    return np.where(ok, num / np.maximum(den, 1e-6), d), valid | ok


def bilateral(d, valid):
    src = np.where(valid, d, -100.0).astype(np.float32)   # 무효 = 동떨어진 값 → 가중치 0
    out = cv2.bilateralFilter(src, P["bilateral_d"], P["bilateral_sigma_disp"], P["bilateral_sigma_px"])
    return np.where(valid, out, d)


def enhance_sequence(log, frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(log.meta)
    meta["enhanced_from"] = os.path.abspath(log.dir)
    meta["enhance"] = P
    meta["frames"] = len(frames)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(log.dir, "index.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fb = log.fx * BASELINE_M                                  # 시차(px) = fb / z
    prev_d, prev_valid, prev_i = None, None, None
    out_rows, t0, stat = [], time.time(), dict(before=0.0, after=0.0)
    for n, i in enumerate(frames):
        z = log.depth_m(i)
        valid = (z > P["z_min"]) & (z < P["z_max"])
        stat["before"] += valid.mean()
        d = np.where(valid, fb / np.where(valid, z, 1), 0).astype(np.float32)
        valid = remove_flying(d, valid)
        d, valid = fill_small_holes(d, valid)
        d = bilateral(d, valid)
        if prev_d is not None and prev_i == i - 1:            # 연속 프레임일 때만 시간 필터
            # 차이가 작을수록 직전 값을 더 섞음 (경계 없는 연속 가중치 → 계단 무늬 없음)
            w = P["temporal_alpha"] * np.exp(-((d - prev_d) / P["temporal_sigma_disp"]) ** 2)
            w = np.where(valid & prev_valid, w, 0).astype(np.float32)
            d = (1 - w) * d + w * prev_d
        prev_d, prev_valid, prev_i = d, valid, i
        zz = np.where(valid, fb / np.where(valid, d, 1), 0)
        raw = np.clip(np.round(zz / log.scale), 0, 65534).astype(np.uint16)
        name = log.index[i][2]
        np.save(os.path.join(out_dir, name), raw)
        stat["after"] += valid.mean()
        r = dict(rows[i])
        r["frame"] = str(n)
        r["valid_ratio"] = f"{valid.mean():.4f}"
        r["median_m"] = f"{np.median(zz[valid]) if valid.any() else 0:.3f}"
        out_rows.append(r)
        if (n + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"[enhance] {n + 1}/{len(frames)} {el:.0f}s eta {el / (n + 1) * (len(frames) - n - 1):.0f}s", flush=True)
    with open(os.path.join(out_dir, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    k = max(len(frames), 1)
    print(f"[enhance] done {len(frames)} frames -> {out_dir}  valid {stat['before'] / k * 100:.1f}% -> "
          f"{stat['after'] / k * 100:.1f}%  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="미포함")
    ap.add_argument("--out", default=None, help="기본: <원본폴더>_enhanced (구간이면 _enhanced_<start>_<end>)")
    a = ap.parse_args()
    log = DepthLog(a.log)
    end = min(a.end or len(log), len(log))
    whole = a.start == 0 and end == len(log)
    out = a.out or (log.dir.rstrip("\\/") + ("_enhanced" if whole else f"_enhanced_{a.start:06d}_{end - 1:06d}"))
    if os.path.abspath(out) == os.path.abspath(log.dir):
        raise SystemExit("출력 폴더가 원본과 같습니다 — 원본은 덮어쓰지 않습니다.")
    enhance_sequence(log, list(range(a.start, end)), out)


if __name__ == "__main__":
    main()
