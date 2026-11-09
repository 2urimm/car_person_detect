"""깊이 로그 -> 3D 점구름 재생 영상(mp4).

카메라 뒤쪽 위에서 따라가며 보는 시점으로, 매 프레임을 실제 거리(m) 3D 점으로 그린다.
- 색 = 바닥으로부터 높이 (0~3 m)
- 바닥 격자 1 m 간격, 카메라 기준 거리 링 2/4/6/8/10 m
- 화면 왼쪽 위: 시간·프레임·정면 최근접 장애물 거리 / 오른쪽 위: 원본 깊이 영상

사용 예:
  python depth3d/render_video.py --start 9000 --end 10500     # 1분 구간
  python depth3d/render_video.py                              # 전체 (약 27분 영상)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_LOG, DEFAULT_OUT, DepthLog
from build_map import estimate_level

W, H = 1280, 720
F = 0.75 * W                       # 가상 카메라 초점거리(px)
EYE = np.array([0.0, -2.2, -3.5])  # 깊이 카메라 기준 뒤 3.5 m, 위 2.2 m (수평 좌표, Y 아래)
LOOK = np.array([0.0, 0.0, 6.0])
H_MAX = 3.0                        # 색상 높이 범위(m)


def look_at(eye, target):
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0, 1.0, 0]), z)    # Y 아래 좌표계: x = down × forward
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])                # 행 = 가상카메라 축 (world -> view 회전)


RV = look_at(EYE, LOOK)


def project(P):
    V = (P - EYE) @ RV.T
    z = V[:, 2]
    ok = z > 0.1
    u = F * V[:, 0] / np.where(ok, z, 1) + W / 2
    v = F * V[:, 1] / np.where(ok, z, 1) + H / 2
    return u, v, z, ok


def polyline(img, pts, color, thick=1):
    u, v, z, ok = project(pts)
    seg = []
    for a in range(len(pts)):
        if ok[a] and -5000 < u[a] < 5000 and -5000 < v[a] < 5000:
            seg.append((int(u[a]), int(v[a])))
        elif seg:
            cv2.polylines(img, [np.array(seg)], False, color, thick, cv2.LINE_AA)
            seg = []
    if len(seg) > 1:
        cv2.polylines(img, [np.array(seg)], False, color, thick, cv2.LINE_AA)


def make_background(cam_h):
    bg = np.full((H, W, 3), 18, np.uint8)
    g = cam_h                                   # 바닥 평면 y (Y 아래 → +cam_h)
    for x in range(-10, 11):
        polyline(bg, np.array([[x, g, z] for z in np.linspace(0, 16, 60)]), (55, 55, 55))
    for z in range(0, 17):
        polyline(bg, np.array([[x, g, z] for x in np.linspace(-10, 10, 60)]), (55, 55, 55))
    for r in (2, 4, 6, 8, 10):
        th = np.linspace(-np.pi / 2, np.pi / 2, 90)
        polyline(bg, np.stack([r * np.sin(th), np.full_like(th, g), r * np.cos(th)], 1), (90, 90, 130), 1)
        u, v, _, ok = project(np.array([[0.0, g, r]]))
        if ok[0]:
            cv2.putText(bg, f"{r} m", (int(u[0]) + 4, int(v[0]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (150, 150, 200), 1, cv2.LINE_AA)
    # 깊이 카메라 위치 표시
    u, v, _, _ = project(np.array([[0.0, 0.0, 0.0]]))
    cv2.drawMarker(bg, (int(u[0]), int(v[0])), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return bg


LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(256, 3)


def render(log, i, R, cam_h, bg, stride):
    P = log.points(i, stride=stride, z_max=15.0) @ R.T
    hgt = cam_h - P[:, 1]                       # 바닥 기준 높이(m)
    u, v, z, ok = project(P)
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z, hgt = u[ok].astype(np.int32), v[ok].astype(np.int32), z[ok], hgt[ok]
    order = np.argsort(-z)                      # 먼 점부터 → 가까운 점이 위에 덮임
    u, v, hgt = u[order], v[order], hgt[order]
    col = LUT[(np.clip(hgt / H_MAX, 0, 1) * 255).astype(np.uint8)]
    img = bg.copy()
    for du in (0, 1):                           # 2x2 점
        for dv in (0, 1):
            img[v + dv, u + du] = col
    # 정면 최근접 장애물: 폭 ±0.5 m, 높이 0.15~2 m
    m = (np.abs(P[:, 0]) < 0.5) & (cam_h - P[:, 1] > 0.15) & (cam_h - P[:, 1] < 2.0)
    near = float(P[m, 2].min()) if m.any() else np.nan
    return img, near


def depth_inset(log, i, w=320):
    d = log.depth_m(i)
    vis = cv2.applyColorMap((np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[d <= 0] = 0
    return cv2.resize(vis, (w, int(w * d.shape[0] / d.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="미포함")
    ap.add_argument("--step", type=int, default=1, help="몇 프레임마다 그릴지 (2면 2배속)")
    ap.add_argument("--stride", type=int, default=2, help="픽셀 간격 (1=전체 해상도, 느림)")
    ap.add_argument("--fps", type=float, default=None, help="기본: 로그 실제 속도에 맞춤")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log = DepthLog(a.log)
    end = min(a.end or len(log), len(log))
    frames = range(a.start, end, a.step)
    fps = a.fps or (end - a.start) / max(log.time_s(end - 1) - log.time_s(a.start), 1e-3) / a.step
    out = a.out or os.path.join(DEFAULT_OUT, f"playback_{a.start:06d}_{end - 1:06d}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    R, cam_h, _ = estimate_level(log, a.start, end)
    if not np.isfinite(cam_h):
        cam_h = 0.5
    print(f"[video] {len(frames)} frames, {fps:.1f} fps, camera height {cam_h:.2f} m -> {out}", flush=True)

    bg = make_background(cam_h)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter 열기 실패: " + out)
    t0 = time.time()
    for n, i in enumerate(frames):
        img, near = render(log, i, R, cam_h, bg, a.stride)
        ins = depth_inset(log, i)
        img[10:10 + ins.shape[0], W - 10 - ins.shape[1]:W - 10] = ins
        cv2.rectangle(img, (W - 11, 9), (W - 10 - ins.shape[1] - 1, 10 + ins.shape[0]), (200, 200, 200), 1)
        t = log.time_s(i)
        txt = [f"t = {int(t // 60):02d}:{t % 60:05.2f}   frame {i}",
               f"front obstacle: {near:.2f} m" if np.isfinite(near) else "front obstacle: -"]
        for k, s in enumerate(txt):
            cv2.putText(img, s, (14, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # 높이 색상 범례
        bar = LUT[np.linspace(255, 0, 150).astype(np.uint8)][:, None, :].repeat(14, 1)
        img[H - 175:H - 25, 20:34] = bar
        cv2.putText(img, f"{H_MAX:.0f} m", (40, H - 168), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, "0 m (ground)", (40, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        vw.write(img)
        if (n + 1) % 500 == 0:
            el = time.time() - t0
            print(f"[video] {n + 1}/{len(frames)} {el:.0f}s eta {el / (n + 1) * (len(frames) - n - 1):.0f}s", flush=True)
    vw.release()
    print(f"[video] done {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
