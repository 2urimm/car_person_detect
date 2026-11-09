"""TSDF 융합 → 매끈한 표면 메시 지도 (원본/보정본 깊이 로그 + build_map.py 의 poses.csv 사용).

여러 프레임의 깊이를 같은 공간 격자(복셀)에 '표면까지의 부호 거리(TSDF)'로 가중 평균하면
프레임마다 다른 스테레오 노이즈가 평균되어 사라지고, 구멍도 다른 프레임 관측으로 메워진다.
그 0-등위면을 Surface Nets 로 삼각형 메시로 뽑는다(외부 라이브러리 없이 numpy 만 사용).

사용 예:
  python depth3d/build_mesh.py --start 9000 --end 10500 \
      --log logs/depth_20261016_195111_enhanced_009000_010499
  python depth3d/build_mesh.py --keyframes            # 전체 경로 (키프레임만, 오래 걸림)
출력: logs/depth3d_out/mesh_<start>_<end>/ (map_mesh.ply, poses.csv, level.json, mesh_preview.png)
→ 뷰어: viewer.html?map=../logs/depth3d_out/mesh_<start>_<end>
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
from common import DEFAULT_LOG, DEFAULT_OUT, DepthLog

OFF = 1 << 20
SX, SY = 1 << 42, 1 << 21          # 키에서 x, y 한 칸 = 이 값만큼 증감 (z 는 1)


def pack(ijk):
    k = ijk.astype(np.int64) + OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack(key):
    m = (1 << 21) - 1
    return np.stack([(key >> 42) & m, (key >> 21) & m, key & m], 1) - OFF


class TSDF:
    def __init__(self, voxel, trunc):
        self.v, self.tr = voxel, trunc
        self.keys = np.zeros(0, np.int64)
        self.ws = np.zeros(0)          # Σ w·sdf
        self.w = np.zeros(0)           # Σ w
        self.nf = np.zeros(0)          # 관측한 프레임 수
        self.pend = []

    def integrate(self, log, i, T, stride, z_max):
        """프레임 i (world <- camera = T)를 누적. 광선 위 표면 앞뒤 ±trunc 구간을 샘플링."""
        z = log.depth_m(i)[::stride, ::stride]
        xn, yn = log._uv(stride)
        ok = (z > 0.3) & (z < z_max)
        z, xn, yn = z[ok], xn[ok], yn[ok]
        n_s = int(np.ceil(2 * self.tr / (self.v * 0.7))) + 1
        offs = np.linspace(-self.tr, self.tr, n_s, dtype=np.float32)
        zs = z[:, None] + offs[None, :]                         # (N, n_s) 샘플 깊이
        Pc = np.stack([xn[:, None] * zs, yn[:, None] * zs, zs], -1).reshape(-1, 3)
        sdf = np.broadcast_to(-offs[None, :] / self.tr, zs.shape).reshape(-1)   # 표면 앞 = +
        # 스테레오 오차 ∝ z² → 먼 관측일수록 가중치를 낮춤
        w = np.broadcast_to((1.0 / np.maximum(z, 1.0) ** 2)[:, None], zs.shape).reshape(-1)
        Pw = Pc @ T[:3, :3].T + T[:3, 3]
        key = pack(np.floor(Pw / self.v))
        uk, inv = np.unique(key, return_inverse=True)
        self.pend.append((uk, np.bincount(inv, w * sdf), np.bincount(inv, w), np.ones(len(uk))))
        if len(self.pend) >= 25:
            self.merge()

    def merge(self):
        if not self.pend:
            return
        k = np.concatenate([self.keys] + [p[0] for p in self.pend])
        a = np.concatenate([self.ws] + [p[1] for p in self.pend])
        b = np.concatenate([self.w] + [p[2] for p in self.pend])
        c = np.concatenate([self.nf] + [p[3] for p in self.pend])
        self.keys, inv = np.unique(k, return_inverse=True)
        self.ws, self.w, self.nf = np.bincount(inv, a), np.bincount(inv, b), np.bincount(inv, c)
        self.pend = []

    def extract(self, w_min, nf_min):
        """Surface Nets: 부호가 바뀌는 셀마다 꼭짓점 1개, 부호가 바뀌는 모서리마다 사각형 1개.
        한두 프레임에서만 보인 복셀(일시적 노이즈)은 제외."""
        self.merge()
        keep = (self.w >= w_min) & (self.nf >= nf_min)
        K, F = self.keys[keep], (self.ws[keep] / self.w[keep]).astype(np.float32)
        order = np.argsort(K)
        K, F = K[order], F[order]

        def look(q):
            j = np.searchsorted(K, q)
            j = np.minimum(j, len(K) - 1)
            hit = K[j] == q
            return np.where(hit, j, -1)

        corners = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        cj = np.stack([look(K + dx * SX + dy * SY + dz) for dx, dy, dz in corners], 1)   # (n, 8)
        full = (cj >= 0).all(1)
        cells = np.nonzero(full)[0]
        cf = F[cj[cells]]                                              # (m, 8) 코너 TSDF
        cross = (cf > 0).any(1) & (cf < 0).any(1)
        cells, cf = cells[cross], cf[cross]
        # 꼭짓점 위치: 부호가 바뀌는 셀 모서리 교차점들의 평균 (셀 내부 좌표 0~1)
        cpos = np.array(corners, np.float32)
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(cpos[a] - cpos[b]).sum() == 1]
        acc = np.zeros((len(cells), 3), np.float32)
        cnt = np.zeros(len(cells), np.float32)
        for a, b in edges:
            fa, fb = cf[:, a], cf[:, b]
            m = (fa > 0) != (fb > 0)
            t = np.where(m, fa / np.where(m, fa - fb, 1), 0)
            acc += m[:, None] * (cpos[a] + t[:, None] * (cpos[b] - cpos[a]))
            cnt += m
        local = acc / np.maximum(cnt, 1)[:, None]
        verts = (unpack(K[cells]).astype(np.float32) + local + 0.5) * self.v   # 복셀 중심 기준
        # 셀 키 -> 꼭짓점 번호
        cell_keys = K[cells]
        vid_of = lambda q: (lambda j: np.where((j < len(cell_keys)) & (cell_keys[np.minimum(j, len(cell_keys) - 1)] == q), j, -1))(np.searchsorted(cell_keys, q))
        faces = []
        # 각 축 방향 모서리 (a, a+e): 그 모서리를 공유하는 4개 셀 (원점이 a, a-u, a-w, a-u-w)
        for e, u, w in ((SX, SY, 1), (SY, 1, SX), (1, SX, SY)):
            j1 = look(K + e)
            m = j1 >= 0
            a_idx = np.nonzero(m)[0]
            fa, fb = F[a_idx], F[j1[m]]
            sc = (fa > 0) != (fb > 0)
            a_idx, fa = a_idx[sc], fa[sc]
            ka = K[a_idx]
            q = np.stack([vid_of(ka), vid_of(ka - u), vid_of(ka - u - w), vid_of(ka - w)], 1)
            good = (q >= 0).all(1)
            q, flip = q[good], fa[good] > 0
            q[flip] = q[flip][:, ::-1]                                  # 앞면 방향 통일
            faces.append(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
        faces = np.concatenate(faces) if faces else np.zeros((0, 3), np.int64)
        return verts, faces


REAL = [(0.0, (0.42, 0.41, 0.39)), (0.12, (0.45, 0.43, 0.40)), (0.25, (0.36, 0.44, 0.30)),
        (1.2, (0.38, 0.47, 0.32)), (2.2, (0.55, 0.53, 0.50)), (4.0, (0.70, 0.69, 0.67))]


def real_colors(h):
    hs = np.array([r[0] for r in REAL])
    cs = np.array([r[1] for r in REAL])
    out = np.stack([np.interp(h, hs, cs[:, c]) for c in range(3)], 1)
    return (out * 255).astype(np.uint8)


def write_mesh_ply(path, V, Fc, C):
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(Fc)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        rec = np.empty(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = V, C
        fh.write(rec.tobytes())
        fr = np.empty(len(Fc), dtype=[("n", "u1"), ("i", "<i4", 3)])
        fr["n"], fr["i"] = 3, Fc
        fh.write(fr.tobytes())


def vertex_normals(V, Fc):
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, Fc[:, k], fn)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def render_mesh(V, Fc, C, eye, target, W=1600, H=1000, fov_deg=70, fog=(12, 52)):
    """소프트웨어 렌더: 삼각형마다 면 위 샘플점을 거리에 맞는 크기로 z-버퍼에 찍음 + 램버트 음영 + 안개."""
    N = vertex_normals(V, Fc)
    L = np.array([0.4, -0.8, 0.3]); L /= np.linalg.norm(L)         # 위(-Y)에서 비스듬히 비추는 빛
    shade = np.clip(np.abs(N @ L), 0, 1) * 0.7 + 0.3
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    x = np.cross(np.array([0, 1.0, 0]), fwd); x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    Rv = np.stack([x, y, fwd])
    f = (W / 2) / np.tan(np.radians(fov_deg / 2))
    # 삼각형 위 샘플: 꼭짓점 3 + 변 중점 3 + 무게중심 (바리센트릭)
    bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]], np.float32)
    zbuf = np.full(H * W, np.inf, np.float32)
    img = np.zeros((H * W, 3), np.float32)
    base = C.astype(np.float32) * shade[:, None]
    for s0 in range(0, len(Fc), 400000):                             # 메모리 절약을 위해 나눠서
        Fb = Fc[s0:s0 + 400000]
        P = np.einsum("sk,fkd->fsd", bary, V[Fb]).reshape(-1, 3)
        Cs = np.einsum("sk,fkd->fsd", bary, base[Fb]).reshape(-1, 3)
        Pv = (P - eye) @ Rv.T
        z = Pv[:, 2]
        ok = z > 0.2
        u = f * Pv[:, 0] / np.where(ok, z, 1) + W / 2
        v = f * Pv[:, 1] / np.where(ok, z, 1) + H / 2
        rad = np.clip(np.ceil(f * 0.035 / np.where(ok, z, 1)), 1, 8).astype(np.int32)   # 면 크기만큼 칠함
        for r in np.unique(rad[ok]):
            m = ok & (rad == r)
            for du in range(-(r // 2), r - r // 2):
                for dv in range(-(r // 2), r - r // 2):
                    uu, vv = (u[m] + du).astype(np.int32), (v[m] + dv).astype(np.int32)
                    inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                    pix = vv[inb] * W + uu[inb]
                    zz, cc = z[m][inb], Cs[m][inb]
                    o = np.argsort(-zz)                                  # 먼 것부터 → 가까운 게 덮음
                    pix, zz, cc = pix[o], zz[o], cc[o]
                    closer = zz < zbuf[pix]
                    zbuf[pix[closer]] = zz[closer]
                    img[pix[closer]] = cc[closer]
    bg = np.array([34, 31, 26], np.float32)                           # RGB
    fog = (np.clip((zbuf - fog[0]) / (fog[1] - fog[0]), 0, 1) if fog else np.zeros_like(zbuf))[:, None]
    hit = np.isfinite(zbuf)[:, None]
    out = np.where(hit, img * (1 - fog) + bg * fog, bg)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8).reshape(H, W, 3)[:, :, ::-1])   # BGR


def preview(path, V, Fc, C, traj):
    """위에서 본 전체 모습 + 경로 중간 지점에서 길 위 눈높이로 본 모습을 한 장에."""
    import cv2
    c = traj.mean(0)
    span = max(np.ptp(traj[:, 0]), np.ptp(traj[:, 2]), 10)
    over = render_mesh(V, Fc, C, c + np.array([0, -span * 0.8, -span * 0.6]), c, 1200, 800, 60, fog=None)
    k = len(traj) // 2
    d = traj[min(k + 5, len(traj) - 1)] - traj[max(k - 5, 0)]
    d[1] = 0
    d = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 1e-3 else np.array([0, 0, 1.0])
    eye = traj[k] - d * 2.0 + np.array([0, -1.2, 0])                  # 카메라 2 m 뒤, 위로 1.2 m
    street = render_mesh(V, Fc, C, eye, eye + d * 10 + np.array([0, 0.9, 0]), 1200, 800, 75)
    for im, t in ((over, "overview"), (street, "street view")):
        cv2.putText(im, t, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imencode(".png", np.hstack([over, street]))[1].tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG, help="깊이 로그 (보정본 폴더도 가능)")
    ap.add_argument("--poses", default=os.path.join(DEFAULT_OUT, "full"), help="poses.csv/level.json 있는 폴더")
    ap.add_argument("--start", type=int, default=0, help="원본 프레임 번호")
    ap.add_argument("--end", type=int, default=None, help="미포함")
    ap.add_argument("--keyframes", action="store_true", help="키프레임 포즈만 사용 (전체 경로용)")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--trunc", type=float, default=0.15)
    ap.add_argument("--zmax", type=float, default=6.0, help="융합에 쓸 최대 거리 (먼 관측은 노이즈 큼)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--wmin", type=float, default=0.6, help="이 가중치 미만 복셀은 표면 추출에서 제외")
    ap.add_argument("--nfmin", type=int, default=3, help="이 프레임 수 미만으로 관측된 복셀 제외")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    log = DepthLog(a.log)
    by_no = {int(f[6:12]): k for k, (_, _, f) in enumerate(log.index)}   # 원본 프레임 번호 -> 로그 인덱스
    with open(os.path.join(a.poses, "poses.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    end = a.end if a.end is not None else 10 ** 9
    sel = [r for r in rows if a.start <= int(r["frame"]) < end and int(r["frame"]) in by_no
           and (not a.keyframes or r["keyframe"] == "1")]
    if not sel:
        raise SystemExit("해당 구간에 포즈가 있는 프레임이 없습니다 (로그/포즈 구간 확인).")
    f0, f1 = int(sel[0]["frame"]), int(sel[-1]["frame"])
    name = a.name or f"mesh_{f0:06d}_{f1:06d}"
    out = os.path.join(DEFAULT_OUT, name)
    os.makedirs(out, exist_ok=True)
    print(f"[mesh] {len(sel)} frames ({f0}..{f1}) from {log.dir}  voxel {a.voxel} trunc {a.trunc} -> {out}", flush=True)

    tsdf = TSDF(a.voxel, a.trunc)
    t0 = time.time()
    traj = []
    for n, r in enumerate(sel):
        T = np.eye(4)
        T[:3, :] = np.array([float(r[f"T{i}{j}"]) for i in range(3) for j in range(4)]).reshape(3, 4)
        tsdf.integrate(log, by_no[int(r["frame"])], T, a.stride, a.zmax)
        traj.append(T[:3, 3])
        if (n + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[mesh] {n + 1}/{len(sel)} voxels~{len(tsdf.keys):,} {el:.0f}s eta {el / (n + 1) * (len(sel) - n - 1):.0f}s", flush=True)
    V, Fc = tsdf.extract(a.wmin, a.nfmin)
    print(f"[mesh] {len(V):,} vertices, {len(Fc):,} triangles ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(a.poses, "level.json"), encoding="utf-8") as fh:
        level = json.load(fh)
    cam_h = level["camera_height_m"]
    # world 는 수평 맞춘 좌표 (Y 아래) → 바닥 기준 높이 = cam_h - y
    C = real_colors(cam_h - V[:, 1])
    write_mesh_ply(os.path.join(out, "map_mesh.ply"), V, Fc, C)
    shutil.copyfile(os.path.join(a.poses, "level.json"), os.path.join(out, "level.json"))
    with open(os.path.join(out, "poses.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(sel)
    preview(os.path.join(out, "mesh_preview.png"), V, Fc, C, np.array(traj))
    with open(os.path.join(out, "mesh_info.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(log=os.path.abspath(log.dir), poses=os.path.abspath(a.poses), frames=len(sel),
                       first=f0, last=f1, voxel=a.voxel, trunc=a.trunc, zmax=a.zmax, wmin=a.wmin, nfmin=a.nfmin,
                       vertices=len(V), triangles=len(Fc)), fh, ensure_ascii=False, indent=1)
    print(f"[mesh] done {out}", flush=True)


if __name__ == "__main__":
    main()
