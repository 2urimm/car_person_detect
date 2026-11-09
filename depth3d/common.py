"""RealSense depth 로그(.npy + meta.json + index.csv) -> 실제 거리(m) 3D 변환 공통 함수.

카메라 좌표계(RealSense 규약): X 오른쪽, Y 아래, Z 앞. 단위 m.
"""
import csv
import json
import os

import numpy as np

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", "depth_20261016_195111")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "logs", "depth3d_out")


class DepthLog:
    def __init__(self, log_dir=DEFAULT_LOG):
        self.dir = log_dir
        with open(os.path.join(log_dir, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        k = self.meta["intrinsics"]
        self.w, self.h = k["width"], k["height"]
        self.fx, self.fy, self.cx, self.cy = k["fx"], k["fy"], k["ppx"], k["ppy"]
        self.scale = self.meta["depth_scale_m"]
        # index.csv: frame, frame_number, timestamp_ms, file, valid_ratio, median_m
        self.index = []
        with open(os.path.join(log_dir, "index.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.index.append((int(row["frame"]), float(row["timestamp_ms"]), row["file"]))
        self.t0 = self.index[0][1]
        self._grid = {}

    def __len__(self):
        return len(self.index)

    def time_s(self, i):
        return (self.index[i][1] - self.t0) / 1000.0

    def depth_m(self, i):
        raw = np.load(os.path.join(self.dir, self.index[i][2]), allow_pickle=False)
        return raw.astype(np.float32) * self.scale

    def _uv(self, stride):
        if stride not in self._grid:
            v, u = np.mgrid[0:self.h:stride, 0:self.w:stride].astype(np.float32)
            self._grid[stride] = ((u - self.cx) / self.fx, (v - self.cy) / self.fy)
        return self._grid[stride]

    def organized_points(self, i, stride=1, z_min=0.2, z_max=15.0):
        """(H/stride, W/stride, 3) 점 배열 + 유효 마스크. 무효 픽셀은 NaN."""
        z = self.depth_m(i)[::stride, ::stride]
        xn, yn = self._uv(stride)
        valid = (z > z_min) & (z < z_max)  # 0=무효, 65.535m(포화) 등 제외
        z = np.where(valid, z, np.nan)
        return np.stack([xn * z, yn * z, z], -1), valid

    def points(self, i, stride=1, z_min=0.2, z_max=15.0):
        P, valid = self.organized_points(i, stride, z_min, z_max)
        return P[valid]


def organized_normals(P):
    """정렬된 점 격자(H,W,3)에서 이웃 차분 외적으로 법선 계산. 계산 불가 픽셀은 NaN."""
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # 카메라 쪽을 향하도록 방향 통일
    flip = np.sum(n * P, -1) > 0
    n[flip] *= -1
    # 깊이 불연속(물체 경계) 근처 법선은 신뢰하지 않음
    span = np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1))
    bad = ~(span < 0.15 * np.maximum(P[..., 2], 0.5))
    n[bad] = np.nan
    return n


def voxel_downsample(P, voxel, *extra):
    """voxel 크기 격자당 첫 점 하나만 남김. extra 배열들도 같은 인덱스로 추림."""
    k = np.floor(P / voxel).astype(np.int64) + (1 << 20)   # 음수 방지 (±1e6 복셀)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, idx = np.unique(key, return_index=True)
    return (P[idx],) + tuple(e[idx] for e in extra)


def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def write_ply(path, P, colors=None):
    """binary little-endian PLY. colors: (N,3) uint8 또는 None."""
    P = np.asarray(P, dtype="<f4")
    with open(path, "wb") as fh:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(P)}",
               "property float x", "property float y", "property float z"]
        if colors is not None:
            hdr += ["property uchar red", "property uchar green", "property uchar blue"]
        fh.write(("\n".join(hdr + ["end_header"]) + "\n").encode())
        if colors is None:
            fh.write(P.tobytes())
        else:
            rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
            rec["p"], rec["c"] = P, colors
            fh.write(rec.tobytes())


def height_colors(h, lo, hi):
    """높이 값 -> turbo 컬러(uint8 RGB)."""
    import cv2
    t = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
    bgr = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return bgr.reshape(-1, 3)[:, ::-1].copy()
