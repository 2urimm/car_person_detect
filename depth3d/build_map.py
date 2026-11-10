"""깊이 로그만으로 카메라 이동 추정(point-to-plane ICP 오도메트리) + 누적 3D 지도 생성.

카메라 위치/IMU 기록이 없으므로 연속 프레임의 형상 정합으로 움직임을 추정한다.
→ 길게 이을수록 오차(드리프트)가 쌓인다. 바닥만 보이거나 긴 복도처럼 형상 특징이 적으면
  전진 방향 이동이 잘 안 잡힐 수 있다(degenerate 열로 표시).

사용 예:
  python depth3d/build_map.py                         # 전체 로그
  python depth3d/build_map.py --start 9000 --end 10500 # 일부 구간
출력 (기본 logs/depth3d_out/<이름>/):
  poses.csv   프레임별 카메라 자세(world<-camera 4x4 앞 3행) + ICP 품질
  map.ply     누적 점구름 (높이 색상, CloudCompare/MeshLab/뷰어에서 열기)
  map_top.png / map_persp.png  미리보기
"""
import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_LOG, DEFAULT_OUT, DepthLog, height_colors, organized_normals,
                    transform, voxel_downsample, write_ply)

ICP_ZMAX = 10.0      # ICP에 쓰는 최대 거리 (먼 점은 노이즈가 커서 제외)
MAP_ZMAX = 8.0       # 지도에 넣는 최대 거리


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def se3_exp(xi):
    """xi = (wx, wy, wz, tx, ty, tz) -> 4x4."""
    w, t = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    K = skew(w / th) if th > 1e-12 else np.zeros((3, 3))
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


R_LEVEL = np.eye(3)  # 카메라 -> 수평 맞춘 카메라 좌표 (main 에서 바닥 평면으로 설정)


class Target:
    """키프레임: 점 + 법선 + KD-tree (수평 맞춘 카메라 좌표)."""

    def __init__(self, log, i):
        P, valid = log.organized_points(i, stride=2, z_max=ICP_ZMAX)
        N = organized_normals(P)
        ok = valid & np.isfinite(N).all(-1)
        P, N = voxel_downsample(P[ok] @ R_LEVEL.T, 0.04, N[ok] @ R_LEVEL.T)
        self.i, self.P, self.N, self.tree = i, P, N, cKDTree(P)


def source_points(log, i):
    P = log.points(i, stride=2, z_max=ICP_ZMAX) @ R_LEVEL.T
    (P,) = voxel_downsample(P, 0.08)
    return P


PLANAR_DOF = [1, 3, 5]   # (wy, tx, tz): 세로축 회전 + 좌우/전후 이동만


def icp(src, tgt, T_init, planar=True, schedule=(0.5,) * 3 + (0.25,) * 4 + (0.1,) * 6):
    """src 를 tgt 좌표로 옮기는 T (tgt <- src). point-to-plane, Huber 가중.
    planar=True 면 바닥 위 평면 운동(3자유도)만 추정 → 높이/기울기 드리프트 없음."""
    T = T_init.copy()
    dof = PLANAR_DOF if planar else list(range(6))
    info = dict(fitness=0.0, rmse=np.nan, cond=np.nan)
    for max_d in schedule:
        p = transform(T, src)
        d, j = tgt.tree.query(p, distance_upper_bound=max_d, workers=-1)
        m = np.isfinite(d)
        if m.sum() < 200:
            break
        p, q, n = p[m], tgt.P[j[m]], tgt.N[j[m]]
        r = np.sum((p - q) * n, 1)
        J = np.hstack([np.cross(p, n), n])[:, dof]
        k = 0.05
        w = k / np.maximum(np.abs(r), k)
        A = J.T @ (J * w[:, None])
        b = -J.T @ (w * r)
        A += np.eye(len(dof)) * 1e-6 * np.trace(A)
        x = np.linalg.solve(A, b)
        xi = np.zeros(6)
        xi[dof] = x
        T = se3_exp(xi) @ T
        ev = np.linalg.eigvalsh(A)
        info = dict(fitness=m.mean(), rmse=float(np.sqrt(np.mean(r ** 2))), cond=float(ev[0] / ev[-1]))
        if np.linalg.norm(x) < 1e-5:
            break
    return T, info


def rotation_to_up(n):
    """바닥 법선 n(위쪽 = -Y 방향) -> (0,-1,0) 으로 보내는 회전."""
    up = np.array([0, -1.0, 0])
    v, c = np.cross(n, up), n @ up
    return np.eye(3) + skew(v) + skew(v) @ skew(v) / (1 + c)


def ground_fit(log, i):
    """프레임 i 의 바닥 평면 RANSAC -> (법선, 카메라 높이, 인라이어 비율) 또는 None."""
    P = log.points(i, stride=2, z_max=8.0)
    low = P[P[:, 1] > 0.1]
    if len(low) < 2000:
        return None
    rng = np.random.default_rng(i)
    best, best_n = None, 0
    for _ in range(300):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[1] > 0:
            n = -n
        if n[1] > -0.7:                  # 대략 수평면만 (±45도)
            continue
        cnt = (np.abs((low - s[0]) @ n) < 0.03).sum()
        if cnt > best_n:
            best_n, best = cnt, (n, s[0])
    if best is None:
        return None
    n, p0 = best
    return n, float(abs(p0 @ n)), best_n / len(P)


def estimate_level(log, start=0, end=None, samples=40):
    """구간 전체에서 고르게 뽑은 프레임들의 바닥 평면으로 카메라 기울기·높이 추정.
    (카메라는 차체에 고정 → 일정하다고 가정) 가려진 프레임이나 바닥이 안 보이는 프레임은 걸러냄."""
    end = end or len(log)
    fits = [ground_fit(log, int(i)) for i in np.linspace(start, end - 1, samples)]
    fits = [f for f in fits if f and 0.15 < f[1] < 2.5 and f[2] > 0.15]
    if len(fits) < 3:
        return np.eye(3), np.nan, 0
    N = np.array([f[0] for f in fits])
    med = np.median(N, 0)
    med /= np.linalg.norm(med)
    agree = [f for f in fits if np.degrees(np.arccos(np.clip(f[0] @ med, -1, 1))) < 4]
    n = np.mean([f[0] for f in agree], 0)
    n /= np.linalg.norm(n)
    return rotation_to_up(n), float(np.median([f[1] for f in agree])), len(agree)


def level_rotation(log, i):
    """(구버전 호환) 프레임 하나의 바닥 평면으로 수평 회전. 실패하면 단위행렬."""
    P = log.points(i, stride=2, z_max=8.0)
    low = P[P[:, 1] > 0.2]
    if len(low) < 500:
        return np.eye(3), np.nan
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(400):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[1] > 0:
            n = -n                       # 위쪽(-Y)을 향하게
        if n[1] > -0.8:
            continue
        cnt = (np.abs((low - s[0]) @ n) < 0.03).sum()
        if cnt > best_n:
            best_n, best = cnt, (n, s[0])
    if best is None:
        return np.eye(3), np.nan
    n, p0 = best
    up = np.array([0, -1.0, 0])
    v, c = np.cross(n, up), n @ up
    R = np.eye(3) + skew(v) + skew(v) @ skew(v) / (1 + c)
    return R, float(abs(p0 @ n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="미포함")
    ap.add_argument("--step", type=int, default=3, help="몇 프레임마다 ICP 할지")
    ap.add_argument("--voxel", type=float, default=0.05, help="지도 해상도(m)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    log = DepthLog(a.log)
    end = min(a.end or len(log), len(log))
    frames = list(range(a.start, end, a.step))
    name = a.name or f"map_{a.start:06d}_{end - 1:06d}"
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    print(f"[map] frames {a.start}..{end - 1} step {a.step} -> {len(frames)} ICP, out {out}", flush=True)

    global R_LEVEL
    # 카메라는 차체에 고정 → 바닥 대비 기울기는 일정하다고 보고, 구간 전체에서 고르게 뽑은 프레임으로 추정
    R_LEVEL, cam_h, n_ok = estimate_level(log, a.start, end)
    tilt = np.degrees(np.arccos(np.clip(R_LEVEL[1, 1], -1, 1)))
    print(f"[map] ground fit from {n_ok} sample frames", flush=True)
    with open(os.path.join(out, "level.json"), "w", encoding="utf-8") as fh:   # 뷰어용
        json.dump({"R_level": R_LEVEL.tolist(), "camera_height_m": cam_h}, fh)
    T_world = np.eye(4)                  # world = 첫 프레임의 수평 맞춘 카메라 좌표
    print(f"[map] camera height ~ {cam_h:.2f} m, tilt vs ground {tilt:.1f} deg (ground fit)", flush=True)

    kf = Target(log, frames[0])
    T_kf = T_world.copy()                # world <- keyframe camera
    T_rel_kf = np.eye(4)                 # keyframe <- 현재 camera
    vel = np.zeros(6)                    # 직전 프레임간 상대운동(초당, 로그맵 근사)
    prev_t = log.time_s(frames[0])
    poses, pending = [], []
    acc = {"keys": np.zeros(0, np.int64), "cnt": np.zeros(0, np.int64)}
    t_start = time.time()

    def merge():
        if not pending:
            return
        k = np.concatenate([acc["keys"]] + pending)
        c = np.concatenate([acc["cnt"]] + [np.ones(len(p), np.int64) for p in pending])
        uk, inv = np.unique(k, return_inverse=True)
        acc["keys"], acc["cnt"] = uk, np.bincount(inv, weights=c).astype(np.int64)
        pending.clear()

    def add_to_map(i, T):
        P = log.points(i, stride=4, z_max=MAP_ZMAX) @ R_LEVEL.T
        W = transform(T, P)
        k = np.floor(W / a.voxel).astype(np.int64) + (1 << 20)
        pending.append(np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]))  # 프레임당 1회만 카운트
        if len(pending) >= 100:
            merge()

    poses.append((frames[0], log.time_s(frames[0]), T_world.copy(), 1.0, 0.0, 1.0, 0, 1))
    add_to_map(frames[0], T_world)

    for n_done, i in enumerate(frames[1:], 1):
        t = log.time_s(i)
        dt = max(t - prev_t, 1e-3)
        prev_t = t
        src = source_points(log, i)
        guess = T_rel_kf @ se3_exp(vel * dt)            # 등속 가정 초기값
        T_new, info = icp(src, kf, guess)
        degenerate = int(info["cond"] < 1e-3 or info["fitness"] < 0.3)
        if degenerate:
            # 등속 예측 자체가 틀려서(예: 직전 회전이 계속 이어짐) 정합이 실패했을 수 있음
            # → '그 자리에 멈춰 있음'을 초기값으로 한 번 더 시도
            T_s, info_s = icp(src, kf, T_rel_kf)
            if not (info_s["cond"] < 1e-3 or info_s["fitness"] < 0.3):
                T_new, info, degenerate = T_s, info_s, 0
        if degenerate:
            # 그래도 실패 -> 회전은 멈추고 이동만 서서히 감속하며 이어감
            # (예전처럼 등속을 그대로 유지하면 몇 초 만에 수백 도씩 도는 가짜 회전이 쌓였음)
            vel = vel.copy()
            vel[:3] = 0
            vel[3:] *= 0.9
            T_new = T_rel_kf @ se3_exp(vel * dt)
        # 속도 갱신 (직전 상대 운동)
        step_T = np.linalg.inv(T_rel_kf) @ T_new
        rv = step_T[:3, :3]
        ang = np.arccos(np.clip((np.trace(rv) - 1) / 2, -1, 1))
        axis = np.array([rv[2, 1] - rv[1, 2], rv[0, 2] - rv[2, 0], rv[1, 0] - rv[0, 1]])
        axis = axis / (np.linalg.norm(axis) + 1e-12) * ang
        vel = np.concatenate([axis, step_T[:3, 3]]) / dt
        T_rel_kf = T_new
        T_cam = T_kf @ T_rel_kf
        new_kf = 0
        moved = np.linalg.norm(T_rel_kf[:3, 3])
        turned = np.degrees(np.arccos(np.clip((np.trace(T_rel_kf[:3, :3]) - 1) / 2, -1, 1)))
        if moved > 0.3 or turned > 8 or info["fitness"] < 0.5:
            kf = Target(log, i)
            T_kf, T_rel_kf = T_cam.copy(), np.eye(4)
            new_kf = 1
            add_to_map(i, T_cam)
        poses.append((i, t, T_cam, info["fitness"], info["rmse"], info["cond"], degenerate, new_kf))
        if n_done % 200 == 0 or n_done == len(frames) - 1:
            pos = T_cam[:3, 3]
            el = time.time() - t_start
            print(f"[map] {n_done}/{len(frames) - 1} frame {i} t={t:.0f}s pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                  f"fit={info['fitness']:.2f} rmse={info['rmse']:.3f} degenerate_total={sum(p[6] for p in poses)} "
                  f"{el:.0f}s eta {el / n_done * (len(frames) - 1 - n_done):.0f}s", flush=True)

    # poses.csv
    with open(os.path.join(out, "poses.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s"] + [f"T{r}{c}" for r in range(3) for c in range(4)] +
                    ["fitness", "rmse", "cond", "degenerate", "keyframe"])
        L = np.eye(4)
        L[:3, :3] = R_LEVEL
        for i, t, T, fit, rmse, cond, deg, kfl in poses:
            T = T @ L                    # world <- 원래 카메라 좌표 (npy 를 바로 변환할 수 있게)
            wr.writerow([i, f"{t:.3f}"] + [f"{x:.6f}" for x in T[:3, :].ravel()] +
                        [f"{fit:.3f}", f"{rmse:.4f}", f"{cond:.2e}", deg, kfl])

    # 지도: 두 번 이상(서로 다른 키프레임에서) 관측된 복셀만
    merge()
    key = acc["keys"][acc["cnt"] >= 2]
    mask = (1 << 21) - 1
    ijk = np.stack([(key >> 42) & mask, (key >> 21) & mask, key & mask], 1) - (1 << 20)
    M = (ijk.astype(np.float32) + 0.5) * a.voxel
    h = -M[:, 1]                                       # 위 = -Y
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    write_ply(os.path.join(out, "map.ply"), M, height_colors(h, lo, hi))
    # 브라우저 뷰어용 가벼운 버전 (10cm)
    (Ml, hl) = voxel_downsample(M, max(0.10, a.voxel * 2), h)
    write_ply(os.path.join(out, "map_light.ply"), Ml, height_colors(hl, lo, hi))
    traj = np.array([p[2][:3, 3] for p in poses])
    np.save(os.path.join(out, "trajectory.npy"), traj)
    length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    print(f"[map] voxels {len(M)}  trajectory length {length:.1f} m  keyframes {sum(p[7] for p in poses)}  "
          f"degenerate {sum(p[6] for p in poses)}/{len(poses)}", flush=True)
    render_previews(out, M, h, lo, hi, traj)
    print("[map] done", out, flush=True)


def render_previews(out, M, h, lo, hi, traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    sub = rng.choice(len(M), min(250000, len(M)), replace=False)
    S, hs = M[sub], h[sub]
    order = np.argsort(hs)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(S[order, 0], S[order, 2], c=hs[order], cmap="turbo", vmin=lo, vmax=hi, s=0.2)
    ax.plot(traj[:, 0], traj[:, 2], "k-", lw=1.2, label="camera path")
    ax.plot(traj[0, 0], traj[0, 2], "go", ms=8, label="start")
    ax.plot(traj[-1, 0], traj[-1, 2], "rs", ms=8, label="end")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title("accumulated map - top view (color = height)")
    _save(fig, os.path.join(out, "map_top.png"))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(S[:, 0], S[:, 2], -S[:, 1], c=hs, cmap="turbo", vmin=lo, vmax=hi, s=0.2)
    ax.plot(traj[:, 0], traj[:, 2], -traj[:, 1], "k-", lw=1.5)
    rngs = np.ptp(S, 0)
    ax.set_box_aspect((rngs[0], rngs[2], max(rngs[1], 1)))
    ax.view_init(elev=35, azim=-60)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("up (m)")
    _save(fig, os.path.join(out, "map_persp.png"))


def _save(fig, path):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, dpi=110)
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
