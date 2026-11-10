"""루프 클로저: '출발 위치 = 도착 위치'라는 사실로 누적 경로 오차(드리프트)를 바로잡는다.

build_map.py 의 ICP 오도메트리는 프레임끼리만 맞추므로 오차가 계속 쌓인다(전체 로그에서 끝점이
시작점과 수백 m 어긋남). 여기서는
  1) 마지막 프레임을 로그 초반 프레임들과 ICP 로 맞춰 '끝 카메라가 시작 카메라에 대해 어디 있는지'를 직접 잰다.
  2) 모든 프레임 자세를 노드, 연속 프레임 ICP 결과를 간선으로 하는 2D 포즈 그래프(x, z, 방향)에
     위 측정을 루프 간선으로 추가해 '방향 먼저, 위치 나중' 선형 최소제곱으로 푼다 → 오차가 경로 전체에
     (ICP 가 불안정했던 구간에 더 많이) 나뉜다. --laps 로 몇 바퀴 돌았는지 알려 주면 전체 회전량을 고정한다
     (방향 오차가 180° 가까이 쌓이면 이것 없이는 경로 모양이 정해지지 않음).
  3) 바로잡은 자세로 누적 지도를 다시 만든다.
원본 지도 폴더는 건드리지 않고 새 폴더(기본 <지도>_loop)에 쓴다.

사용 예:
  python depth3d/loop_close.py                          # 끝 프레임 ↔ 초반 3분 중 가장 잘 맞는 프레임 자동 탐색
  python depth3d/loop_close.py --pair 41685:2088        # 직접 지정 (끝프레임:시작프레임)
  python depth3d/loop_close.py --pair 41685:2088 --laps 1   # 반시계로 한 바퀴 돌았음
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_map as bm
from common import DEFAULT_LOG, DEFAULT_OUT, DepthLog, height_colors, transform, voxel_downsample, write_ply


# ---------------------------------------------------------------- 자세 입출력
def load_poses(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    T = np.zeros((len(rows), 4, 4))
    T[:, 3, 3] = 1
    for k, r in enumerate(rows):
        T[k, :3, :] = np.array([float(r[f"T{a}{b}"]) for a in range(3) for b in range(4)]).reshape(3, 4)
    return rows, T


def to_planar(T_lev):
    """수평 맞춘 world<-camera 자세 -> (x, z, yaw). 회전은 세로축(Y) 회전뿐."""
    return np.stack([T_lev[:, 0, 3], T_lev[:, 2, 3], np.arctan2(T_lev[:, 0, 2], T_lev[:, 2, 2])], 1)


def from_planar(X, T_lev_old):
    T = T_lev_old.copy()
    c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
    T[:, 0, 0], T[:, 0, 2], T[:, 2, 0], T[:, 2, 2] = c, s, -s, c
    T[:, 0, 1] = T[:, 1, 0] = T[:, 1, 2] = T[:, 2, 1] = 0
    T[:, 1, 1] = 1
    T[:, 0, 3], T[:, 2, 3] = X[:, 0], X[:, 1]
    return T


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rel(Xi, Xj):
    """노드 i 좌표계에서 본 j (lateral, forward, dyaw)."""
    c, s = np.cos(Xi[..., 2]), np.sin(Xi[..., 2])
    dx, dz = Xj[..., 0] - Xi[..., 0], Xj[..., 1] - Xi[..., 1]
    return np.stack([c * dx - s * dz, s * dx + c * dz, wrap(Xj[..., 2] - Xi[..., 2])], -1)


# ---------------------------------------------------------------- 루프 측정 (ICP)
def match_pair(log, q, c, coarse=True):
    """끝 프레임 q 를 시작쪽 프레임 c 에 맞춤 -> (T c<-q (수평 좌표), info). 초기값을 여러 개 시도."""
    src = bm.source_points(log, q)
    tgt = bm.Target(log, c)
    yaws = range(-180, 180, 20) if coarse else range(-40, 41, 10)
    sched = (0.5, 0.5, 0.25, 0.25, 0.1, 0.1)
    best = None
    for yaw in yaws:
        for tx, tz in ((0, 0), (1.5, 0), (-1.5, 0), (0, 2), (0, -2)):
            T0 = bm.se3_exp(np.array([0, np.radians(yaw), 0, tx, 0, tz]))
            T, info = bm.icp(src, tgt, T0, schedule=sched)
            if best is None or info["fitness"] > best[1]["fitness"]:
                best = (T, info)
    T, info = bm.icp(src, tgt, best[0])            # 최종 정밀 정합
    return T, info


def find_loop(log, q, frames, search_s):
    """frames 중 처음 search_s 초 안에서 q 와 가장 잘 맞는 프레임을 찾는다."""
    t0 = log.time_s(frames[0])
    cands = [f for f in frames if log.time_s(f) - t0 <= search_s][::30]
    best = None
    for c in cands:
        T, info = match_pair(log, q, c)
        print(f"[loop] cand {c:6d} t={log.time_s(c):5.0f}s fitness {info['fitness']:.2f} rmse {info['rmse']:.3f}", flush=True)
        if best is None or info["fitness"] > best[2]["fitness"]:
            best = (c, T, info)
    # 최고 후보 주변을 촘촘히 다시 본다
    c0 = best[0]
    k = frames.index(c0)
    for c in frames[max(0, k - 30):k + 31:6]:
        T, info = match_pair(log, q, c, coarse=False)
        if info["fitness"] > best[2]["fitness"]:
            best = (c, T, info)
    return best


# ---------------------------------------------------------------- 포즈 그래프
def odom_sigmas(d, deg, fit):
    """연속 프레임 간선의 표준편차 (lateral, forward, yaw). 정합이 나빴던 구간일수록 크게."""
    step = np.hypot(d[:, 0], d[:, 1])
    # 멈춰 있을 때는 거의 정확, 많이 움직이거나 돌수록 오차가 커진다고 봄
    s = np.stack([0.005 + 0.02 * step,
                  0.005 + 0.03 * step,
                  np.radians(0.05) + np.radians(0.5) * step + 0.03 * np.abs(d[:, 2])], 1)
    s[(deg == 0) & (fit < 0.45)] *= 2
    s[deg == 1] += np.array([0.05, 0.10, np.radians(3.0)])   # 정합 실패 구간: 거의 모름
    return s


def world_vec(th, v):
    """노드 방향 th 기준 로컬 (lateral, forward) -> world (x, z)."""
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def chain(x0, d, dth):
    """시작 자세 x0 에서 상대 이동 d(로컬 이동)와 방향 변화 dth 로 경로를 다시 잇는다 (방향은 감지 않은 누적값)."""
    th = x0[2] + np.concatenate([[0], np.cumsum(dth)])
    steps = world_vec(th[:-1], d[:, :2])
    p = x0[:2] + np.concatenate([[[0, 0]], np.cumsum(steps, 0)])
    return np.column_stack([p, th])


def close_loop(X0, d, sig, kc, kq, z_loop, laps=None, loop_sig=(0.3, np.radians(2.0))):
    """루프 하나짜리 포즈 그래프를 '방향 먼저, 위치 나중' 두 단계 선형 최소제곱으로 정확히 푼다.
    laps: 시작~끝 사이에 위에서 봤을 때 몇 바퀴 돌았는지 (+ 반시계, - 시계). 주면 전체 회전량을
    z_yaw + 360k 중 -360*laps 에 가장 가까운 값으로 고정한다. 안 주면 가장 작은 보정(±180° 안)을 쓴다.
    (yaw 는 오른쪽으로 돌 때 + → 위에서 봐서 시계방향이 +)"""
    turn_odo = d[kc:kq, 2].sum()                       # 오도메트리가 말하는 누적 회전 (감지 않음)
    if laps is None:
        target = turn_odo - wrap(turn_odo - z_loop[2])
    else:
        k = np.round((-2 * np.pi * laps - z_loop[2]) / (2 * np.pi))
        target = z_loop[2] + 2 * np.pi * k
    # 1) 방향: 오차를 간선별 yaw 분산 비율로 나눔
    w = np.zeros(len(d))
    w[kc:kq] = sig[kc:kq, 2] ** 2
    e = turn_odo - target
    dth = d[:, 2] - e * w / (w.sum() + loop_sig[1] ** 2)
    X = chain(X0[0], d, dth)
    # 2) 위치: 방향을 고정하면 선형. 간선별 world 공분산 C_k 비율로 위치 오차를 나눔
    th = X[:-1, 2]
    c, s = np.cos(th), np.sin(th)
    Rk = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)       # world <- local
    Dk = np.zeros((len(d), 2, 2))
    Dk[:, 0, 0], Dk[:, 1, 1] = sig[:, 0] ** 2, sig[:, 1] ** 2
    Ck = Rk @ Dk @ np.transpose(Rk, (0, 2, 1))
    Ck[:kc] = 0
    Ck[kq:] = 0
    e_p = (X[kq, :2] - X[kc, :2]) - world_vec(X[kc, 2], z_loop[:2])
    S = Ck.sum(0) + np.eye(2) * loop_sig[0] ** 2
    corr = -(Ck @ np.linalg.solve(S, e_p))                                  # 간선별 world 이동 보정
    steps = np.diff(X[:, :2], axis=0) + corr
    X[1:, :2] = X[0, :2] + np.cumsum(steps, 0)
    info = dict(turn_odometry_deg=float(np.degrees(turn_odo)), turn_target_deg=float(np.degrees(target)),
                heading_error_deg=float(np.degrees(e)), position_error_m=float(np.hypot(*e_p)))
    X[:, 2] = wrap(X[:, 2])
    return X, info


# ---------------------------------------------------------------- 지도 재생성
def rebuild_map(log, frames, T_cam, keyflag, voxel, R_level, out):
    keys = []
    acc_k, acc_c = np.zeros(0, np.int64), np.zeros(0, np.int64)
    sel = np.where(keyflag == 1)[0]
    t0 = time.time()
    for n_done, k in enumerate(sel):
        P = log.points(frames[k], stride=4, z_max=bm.MAP_ZMAX)
        W = transform(T_cam[k], P)                 # T_cam 은 world <- 원래 카메라 좌표
        q = np.floor(W / voxel).astype(np.int64) + (1 << 20)
        keys.append(np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]))
        if len(keys) >= 100 or n_done == len(sel) - 1:
            kk = np.concatenate([acc_k] + keys)
            cc = np.concatenate([acc_c] + [np.ones(len(x), np.int64) for x in keys])
            acc_k, inv = np.unique(kk, return_inverse=True)
            acc_c = np.bincount(inv, weights=cc).astype(np.int64)
            keys = []
            print(f"[map] {n_done + 1}/{len(sel)} keyframes {time.time() - t0:.0f}s", flush=True)
    key = acc_k[acc_c >= 2]
    mask = (1 << 21) - 1
    ijk = np.stack([(key >> 42) & mask, (key >> 21) & mask, key & mask], 1) - (1 << 20)
    M = (ijk.astype(np.float32) + 0.5) * voxel
    h = -M[:, 1]
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    write_ply(os.path.join(out, "map.ply"), M, height_colors(h, lo, hi))
    Ml, hl = voxel_downsample(M, max(0.10, voxel * 2), h)
    write_ply(os.path.join(out, "map_light.ply"), Ml, height_colors(hl, lo, hi))
    return M, h, lo, hi


def compare_plot(path, X_old, X_new, k_start, k_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for ax, X, ttl in ((axs[0], X_old, "before (ICP odometry only)"), (axs[1], X_new, "after loop closure")):
        ax.plot(X[:, 0], X[:, 1], "k-", lw=1)
        ax.plot(*X[k_start, :2], "go", ms=9, label="loop start frame")
        ax.plot(*X[k_end, :2], "rs", ms=9, label="end frame")
        gap = np.hypot(*(X[k_end, :2] - X[k_start, :2]))
        ax.set_title(f"{ttl}\nstart-end gap {gap:.1f} m")
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    bm._save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--map", default=os.path.join(DEFAULT_OUT, "full_v2"), help="build_map.py 결과 폴더")
    ap.add_argument("--out", default=None, help="기본: <map>_loop")
    ap.add_argument("--pair", default=None, help="끝프레임:시작프레임 (생략하면 자동 탐색)")
    ap.add_argument("--search", type=float, default=180, help="자동 탐색 시 로그 처음 몇 초를 볼지")
    ap.add_argument("--min-fitness", type=float, default=0.15)
    ap.add_argument("--laps", type=int, default=None,
                    help="위에서 봤을 때 몇 바퀴 돌았는지 (+1 반시계 한 바퀴, -1 시계 한 바퀴). 전체 회전량을 고정함")
    ap.add_argument("--voxel", type=float, default=0.05)
    a = ap.parse_args()

    out = a.out or a.map.rstrip("/\\") + "_loop"
    if os.path.abspath(out) == os.path.abspath(a.map):
        sys.exit("출력 폴더가 원본 지도 폴더와 같습니다.")
    os.makedirs(out, exist_ok=True)
    log = DepthLog(a.log)
    with open(os.path.join(a.map, "level.json"), encoding="utf-8") as fh:
        level = json.load(fh)
    bm.R_LEVEL = np.array(level["R_level"])
    L = np.eye(4)
    L[:3, :3] = bm.R_LEVEL
    rows, T_cam = load_poses(os.path.join(a.map, "poses.csv"))
    frames = [int(r["frame"]) for r in rows]
    deg = np.array([int(r["degenerate"]) for r in rows])
    fit = np.array([float(r["fitness"]) for r in rows])
    keyflag = np.array([int(r["keyframe"]) for r in rows])
    T_lev = T_cam @ np.linalg.inv(L)               # world <- 수평 맞춘 카메라
    X0 = to_planar(T_lev)

    # 1) 루프 측정
    if a.pair:
        q, c = (int(v) for v in a.pair.split(":"))
        q = frames[int(np.argmin(np.abs(np.array(frames) - q)))]
        c = frames[int(np.argmin(np.abs(np.array(frames) - c)))]
        T_loop, info = match_pair(log, q, c)
    else:
        q = frames[-1]
        c, T_loop, info = find_loop(log, q, frames, a.search)
    yaw = np.degrees(np.arctan2(T_loop[0, 2], T_loop[2, 2]))
    print(f"[loop] end {q} -> start {c}: fitness {info['fitness']:.2f} rmse {info['rmse']:.3f} "
          f"offset ({T_loop[0, 3]:.2f}, {T_loop[2, 3]:.2f}) m yaw {yaw:.1f} deg", flush=True)
    if info["fitness"] < a.min_fitness:
        sys.exit(f"루프 정합이 약합니다(fitness {info['fitness']:.2f}). --pair 로 직접 지정하세요.")
    kq, kc = frames.index(q), frames.index(c)
    z_loop = np.array([T_loop[0, 3], T_loop[2, 3], np.arctan2(T_loop[0, 2], T_loop[2, 2])])
    before = rel(X0[kc], X0[kq])
    print(f"[loop] odometry says end is ({before[0]:.1f}, {before[1]:.1f}) m / {np.degrees(before[2]):.1f} deg "
          f"from start frame; ICP says ({z_loop[0]:.2f}, {z_loop[1]:.2f}) m / {np.degrees(z_loop[2]):.1f} deg", flush=True)

    # 2) 포즈 그래프
    d = rel(X0[:-1], X0[1:])
    sig = odom_sigmas(d, deg[1:], fit[1:])
    X, ginfo = close_loop(X0, d, sig, kc, kq, z_loop, a.laps)   # 루프 측정 오차 0.3 m / 2° 감안
    after = rel(X[kc], X[kq])
    print(f"[graph] total turn: odometry {ginfo['turn_odometry_deg']:.0f} deg -> {ginfo['turn_target_deg']:.0f} deg "
          f"(laps={a.laps}); corrected heading {ginfo['heading_error_deg']:.0f} deg, position {ginfo['position_error_m']:.0f} m",
          flush=True)
    print(f"[graph] loop residual ({after[0] - z_loop[0]:.2f}, {after[1] - z_loop[1]:.2f}) m "
          f"{np.degrees(wrap(after[2] - z_loop[2])):.2f} deg", flush=True)

    T_new = from_planar(X, T_lev) @ L
    with open(os.path.join(out, "poses.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        cols = list(rows[0].keys())
        wr.writerow(cols)
        for r, T in zip(rows, T_new):
            r = dict(r)
            for i_ in range(3):
                for j_ in range(4):
                    r[f"T{i_}{j_}"] = f"{T[i_, j_]:.6f}"
            wr.writerow([r[k] for k in cols])
    shutil.copy(os.path.join(a.map, "level.json"), os.path.join(out, "level.json"))
    traj = T_new[:, :3, 3]
    np.save(os.path.join(out, "trajectory.npy"), traj)
    with open(os.path.join(out, "loop.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_map": os.path.abspath(a.map), "end_frame": q, "start_frame": c,
                   "icp_fitness": info["fitness"], "icp_rmse": info["rmse"],
                   "measured_end_in_start": {"lateral_m": z_loop[0], "forward_m": z_loop[1], "yaw_deg": yaw},
                   "laps": a.laps, **ginfo,
                   "gap_before_m": float(np.hypot(*(X0[kq, :2] - X0[kc, :2]))),
                   "gap_after_m": float(np.hypot(*(X[kq, :2] - X[kc, :2]))),
                   "path_length_m": float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())},
                  fh, ensure_ascii=False, indent=1)
    compare_plot(os.path.join(out, "loop_compare.png"), X0, X, kc, kq)

    # 3) 지도 재생성
    M, h, lo, hi = rebuild_map(log, frames, T_new, keyflag, a.voxel, bm.R_LEVEL, out)
    bm.render_previews(out, M, h, lo, hi, traj)
    print(f"[loop] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
