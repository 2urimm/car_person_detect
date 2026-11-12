"""이미 만든 누적 지도(poses.csv)에서 원하는 시간 구간만 잘라 새 지도를 만든다.

자세는 다시 추정하지 않고 원래 지도(기본: 루프 보정한 full_v2_loop)의 자세를 그대로 쓴다.
→ 잘라낸 구간도 전체 지도와 같은 좌표에 놓인다. 원래 지도 폴더는 건드리지 않는다.

사용 예:
  python depth3d/crop_map.py --start-s 405 --end-s 940 --name art_valley
출력 (logs/depth3d_out/<name>/):
  map.ply, map_light.ply   구간 누적 점구름 (뷰어: viewer.html?map=../logs/depth3d_out/<name>)
  poses.csv, level.json    구간 자세 (뷰어 재생·월드 좌표용)
  map_top.png, map_persp.png  미리보기
  location.png             전체 경로 중 잘라낸 구간 위치
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_map as bm
import loop_close as lc
from common import DEFAULT_LOG, DEFAULT_OUT, DepthLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--map", default=os.path.join(DEFAULT_OUT, "full_v2_loop"), help="자세를 가져올 지도 폴더")
    ap.add_argument("--start-s", type=float, required=True, help="구간 시작 (로그 시작 기준 초)")
    ap.add_argument("--end-s", type=float, required=True, help="구간 끝 (초)")
    ap.add_argument("--name", default=None, help="출력 폴더 이름 (기본 crop_<시작>_<끝>s)")
    ap.add_argument("--voxel", type=float, default=0.05)
    a = ap.parse_args()

    name = a.name or f"crop_{a.start_s:.0f}_{a.end_s:.0f}s"
    out = os.path.join(DEFAULT_OUT, name)
    if os.path.abspath(out) == os.path.abspath(a.map):
        sys.exit("출력 폴더가 원래 지도 폴더와 같습니다.")
    os.makedirs(out, exist_ok=True)

    log = DepthLog(a.log)
    with open(os.path.join(a.map, "level.json"), encoding="utf-8") as fh:
        bm.R_LEVEL = np.array(json.load(fh)["R_level"])
    rows, T = lc.load_poses(os.path.join(a.map, "poses.csv"))
    t = np.array([float(r["time_s"]) for r in rows])
    sel = np.where((t >= a.start_s) & (t <= a.end_s))[0]
    if len(sel) < 2:
        sys.exit("구간에 프레임이 없습니다.")
    frames = [int(rows[k]["frame"]) for k in sel]
    keyflag = np.array([int(rows[k]["keyframe"]) for k in sel])
    keyflag[0] = 1                                   # 구간 첫 프레임은 항상 지도에 넣음
    print(f"[crop] {a.start_s:.0f}-{a.end_s:.0f}s: frames {frames[0]}..{frames[-1]} "
          f"({len(sel)} poses, {keyflag.sum()} keyframes) -> {out}", flush=True)

    M, h, lo, hi = lc.rebuild_map(log, frames, T[sel], keyflag, a.voxel, bm.R_LEVEL, out)
    traj = T[sel, :3, 3]
    bm.render_previews(out, M, h, lo, hi, traj)

    with open(os.path.join(out, "poses.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows[k] for k in sel)
    shutil.copy(os.path.join(a.map, "level.json"), os.path.join(out, "level.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = T[:, :3, 3]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(full[:, 0], full[:, 2], "-", color="0.75", lw=1, label="whole path")
    ax.plot(traj[:, 0], traj[:, 2], "r-", lw=2, label=f"crop {a.start_s:.0f}-{a.end_s:.0f}s")
    ax.plot(*traj[0, [0, 2]], "go", ms=8, label="crop start")
    ax.plot(*traj[-1, [0, 2]], "ks", ms=8, label="crop end")
    for s in range(0, int(t[-1]) + 1, 60):
        k = np.searchsorted(t, s)
        ax.annotate(f"{s // 60}m", full[k, [0, 2]], fontsize=7, color="0.4")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="best")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title("crop location (labels = minutes)")
    bm._save(fig, os.path.join(out, "location.png"))
    length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    print(f"[crop] voxels {len(M)}  path {length:.1f} m  done -> {out}", flush=True)


if __name__ == "__main__":
    main()
