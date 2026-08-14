#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Align a re-exported STL back onto the frame of a reference STL.

CAD exports frequently land in assembly coordinates or with the axes relabelled, so a
freshly exported link mesh no longer sits where the URDF expects it. This tool recovers
the rigid transform that maps the new export onto a reference mesh whose frame is already
trusted, and either prints it as a URDF <origin> or bakes it into a new STL.

The search is restricted to the 24 proper axis-aligned rotations (signed axis
permutations with det = +1), which is what CAD exporters actually produce. Candidates are
filtered by bounding-box dimensions, then ranked by surface voxel IoU to break the
symmetry that the bounding box alone cannot resolve.

Examples
--------
Report the <origin> to paste into the URDF::

    ./align_mesh.py ../meshes/omy_f3m/link6.stl ../meshes/omy_f3m_d435/link6_new.stl

Bake the transform into the mesh instead, so the URDF origin can stay identity::

    ./align_mesh.py ../meshes/omy_f3m/link6.stl ../meshes/omy_f3m_d435/link6_new.stl \
        -o ../meshes/omy_f3m_d435/link6.stl
"""

import argparse
import itertools
import struct
import sys

import numpy as np


# --------------------------------------------------------------------------------------
# STL I/O
# --------------------------------------------------------------------------------------

class Stl:
    """A binary STL, kept as raw records so the header and attribute bytes survive."""

    def __init__(self, header, normals, verts, attrs):
        self.header = header
        self.normals = normals
        self.verts = verts
        self.attrs = attrs

    @property
    def count(self):
        return len(self.verts)

    @property
    def points(self):
        return self.verts.reshape(-1, 3)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            data = handle.read()
        if len(data) < 84:
            raise ValueError(f'{path}: too short to be an STL')

        if data[:5].lower() == b'solid' and b'facet' in data[:2048]:
            return cls._load_ascii(data)

        n = struct.unpack('<I', data[80:84])[0]
        expected = 84 + n * 50
        if len(data) < expected:
            raise ValueError(
                f'{path}: truncated binary STL (need {expected} bytes, have {len(data)})')
        if len(data) > expected:
            print(f'  note: {len(data) - expected} trailing bytes after the triangle '
                  f'data will be dropped', file=sys.stderr)

        rec = np.frombuffer(data[84:expected], dtype=np.uint8).reshape(n, 50)
        return cls(
            header=data[:80],
            normals=rec[:, 0:12].copy().view(np.float32).reshape(n, 3).astype(np.float64),
            verts=rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64),
            attrs=rec[:, 48:50].copy(),
        )

    @classmethod
    def _load_ascii(cls, data):
        normals, verts, cur = [], [], []
        for line in data.decode('utf-8', 'replace').splitlines():
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'facet' and len(tok) >= 5:
                normals.append([float(v) for v in tok[2:5]])
            elif tok[0] == 'vertex':
                cur.append([float(v) for v in tok[1:4]])
                if len(cur) == 3:
                    verts.append(cur)
                    cur = []
        n = len(verts)
        if n == 0:
            raise ValueError('ASCII STL contained no triangles')
        if len(normals) != n:
            normals = [[0.0, 0.0, 0.0]] * n
        return cls(
            header=b'converted from ASCII STL by align_mesh.py'.ljust(80, b'\0'),
            normals=np.array(normals, dtype=np.float64),
            verts=np.array(verts, dtype=np.float64),
            attrs=np.zeros((n, 2), dtype=np.uint8),
        )

    def transformed(self, rot, trans):
        """Return a copy with rot/trans applied to vertices and rot applied to normals."""
        return Stl(self.header, self.normals @ rot.T, self.verts @ rot.T + trans, self.attrs)

    def save(self, path):
        n = self.count
        rec = np.zeros((n, 50), dtype=np.uint8)
        rec[:, 0:12] = self.normals.astype(np.float32).view(np.uint8).reshape(n, 12)
        rec[:, 12:48] = self.verts.astype(np.float32).view(np.uint8).reshape(n, 36)
        rec[:, 48:50] = self.attrs
        with open(path, 'wb') as handle:
            handle.write(self.header)
            handle.write(struct.pack('<I', n))
            handle.write(rec.tobytes())

    def signed_volume(self):
        """Signed volume via the divergence theorem; the sign detects mirroring."""
        v = self.verts
        return float(np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    def flipped_faces(self):
        """Number of faces whose stored normal disagrees with the winding order."""
        v = self.verts
        cr = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        area = np.linalg.norm(cr, axis=1)
        ok = area > 1e-12
        if not ok.any():
            return 0
        dot = np.einsum('ij,ij->i', cr[ok] / area[ok, None], self.normals[ok])
        return int((dot < 0).sum())


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------

def proper_axis_rotations():
    """The 24 rotations a CAD exporter can introduce: signed axis permutations, det=+1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if abs(np.linalg.det(m) - 1.0) < 1e-9:
                out.append(m)
    return out


def surface_samples(verts, per_edge=5):
    """Scatter points over every triangle so thin features still occupy voxels."""
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
    pts = [v0, v1, v2]
    for a in np.linspace(0.0, 1.0, per_edge):
        for b in np.linspace(0.0, 1.0 - a, per_edge):
            pts.append(v0 + a * (v1 - v0) + b * (v2 - v0))
    return np.vstack(pts)


def voxel_iou(pts_a, pts_b, origin, pitch, dims):
    def grid(p):
        idx = np.floor((p - origin) / pitch).astype(int)
        keep = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[keep]
        g = np.zeros(dims, dtype=bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return g

    ga, gb = grid(pts_a), grid(pts_b)
    union = (ga | gb).sum()
    return float((ga & gb).sum() / union) if union else 0.0


def rot_to_rpy(rot):
    """Convert a rotation matrix to URDF fixed-axis rpy, where R = Rz(y) Ry(p) Rx(r)."""
    if abs(rot[2, 0]) < 1.0 - 1e-9:
        pitch = -np.arcsin(rot[2, 0])
        cp = np.cos(pitch)
        roll = np.arctan2(rot[2, 1] / cp, rot[2, 2] / cp)
        yaw = np.arctan2(rot[1, 0] / cp, rot[0, 0] / cp)
    else:
        # Gimbal lock: roll and yaw share an axis, so pin yaw to zero.
        yaw = 0.0
        if rot[2, 0] <= -1.0 + 1e-9:
            pitch = np.pi / 2
            roll = np.arctan2(rot[0, 1], rot[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-rot[0, 1], -rot[0, 2])
    return np.array([roll, pitch, yaw])


def rpy_to_rot(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------

def align(ref, src, pitch, size_tol, verbose=True):
    """Find rot, trans mapping src onto ref. Returns (rot, trans, iou, exact)."""
    ref_pts, src_pts = ref.points, src.points
    ref_lo, ref_hi = ref_pts.min(0), ref_pts.max(0)
    ref_size, ref_mid = ref_hi - ref_lo, (ref_hi + ref_lo) / 2

    origin = ref_lo - 3 * pitch
    dims = np.ceil((ref_hi + 3 * pitch - origin) / pitch).astype(int)
    ref_samples = surface_samples(ref.verts)
    src_samples = surface_samples(src.verts)

    ranked = []
    for rot in proper_axis_rotations():
        rp = src_pts @ rot.T
        size_err = float(np.abs((rp.max(0) - rp.min(0)) - ref_size).sum())
        if size_err > size_tol:
            continue
        trans = ref_mid - (rp.max(0) + rp.min(0)) / 2
        iou = voxel_iou(ref_samples, src_samples @ rot.T + trans, origin, pitch, dims)
        ranked.append((iou, size_err, rot, trans))

    if not ranked:
        raise SystemExit(
            'No axis-aligned rotation matched the reference bounding box.\n'
            'The meshes are probably different parts, or the export is scaled '
            '(check mm vs m), or the rotation is not axis-aligned.')

    ranked.sort(key=lambda c: -c[0])
    iou, size_err, rot, trans = ranked[0]

    if verbose:
        print('  rotation candidates (bbox-filtered, ranked by voxel IoU):')
        for cand_iou, cand_err, cand_rot, _ in ranked[:3]:
            print(f'    IoU={cand_iou:.3f}  size-err={cand_err:6.2f} mm  '
                  f'R={cand_rot.astype(int).tolist()}')
        if len(ranked) > 1 and iou - ranked[1][0] < 0.05:
            print('    WARNING: top two candidates are close; inspect the result visually.')

    # With identical geometry the bbox pins the translation exactly. Cross-check the min
    # and max faces: agreement means the shape is unchanged and this is an exact answer.
    rp = src_pts @ rot.T
    t_lo, t_hi = ref_lo - rp.min(0), ref_hi - rp.max(0)
    spread = float(np.abs(t_hi - t_lo).max())
    exact = spread < 1e-3

    if exact:
        trans = (t_lo + t_hi) / 2
    else:
        if verbose:
            print(f'  bbox faces disagree by {spread:.3f} mm -> the part itself changed; '
                  f'refining translation by IoU search')
        best = (iou, trans)
        for step, span in ((1.0, 5.0), (0.25, 1.5)):
            grid_range = np.arange(-span, span + 1e-9, step)
            base = best[1]
            for dx in grid_range:
                for dy in grid_range:
                    for dz in grid_range:
                        cand = base + np.array([dx, dy, dz])
                        score = voxel_iou(ref_samples, src_samples @ rot.T + cand,
                                          origin, pitch, dims)
                        if score > best[0]:
                            best = (score, cand)
        iou, trans = best

    return rot, trans, iou, exact


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Align a re-exported STL onto the frame of a reference STL.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Examples\n--------\n')[-1])
    ap.add_argument('reference', help='STL already in the desired (URDF link) frame')
    ap.add_argument('input', help='freshly exported STL to align')
    ap.add_argument('-o', '--output',
                    help='bake the transform into this STL instead of only reporting it')
    ap.add_argument('--pitch', type=float, default=2.0,
                    help='voxel size in mm for the overlap test (default: 2.0)')
    ap.add_argument('--size-tol', type=float, default=25.0,
                    help='max summed bbox dimension mismatch, mm (default: 25.0)')
    args = ap.parse_args()

    ref = Stl.load(args.reference)
    src = Stl.load(args.input)
    print(f'reference : {args.reference}  ({ref.count} tris)')
    print(f'  bbox min {np.round(ref.points.min(0), 3)}  max {np.round(ref.points.max(0), 3)}')
    print(f'input     : {args.input}  ({src.count} tris)')
    print(f'  bbox min {np.round(src.points.min(0), 3)}  max {np.round(src.points.max(0), 3)}')
    print()

    rot, trans, iou, exact = align(ref, src, args.pitch, args.size_tol)
    rpy = rot_to_rpy(rot)
    if not np.allclose(rpy_to_rot(rpy), rot, atol=1e-9):
        raise SystemExit('internal error: rpy does not reproduce the rotation matrix')

    out = src.transformed(rot, trans)
    dev = float(np.abs(np.concatenate([
        out.points.min(0) - ref.points.min(0),
        out.points.max(0) - ref.points.max(0)])).max())

    print(f'\n  overlap IoU        : {iou:.4f}')
    print(f'  translation        : {"exact (from bbox)" if exact else "IoU-refined"}')
    print(f'  bbox deviation     : {dev:.6f} mm')

    if args.output is None:
        print('\nPaste into the link <visual>/<collision>:\n')
        print(f'  <origin xyz="{trans[0] / 1000:.6f} {trans[1] / 1000:.6f} '
              f'{trans[2] / 1000:.6f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}" />')
        print('\n(xyz is in metres, matching scale="0.001 0.001 0.001" on the mesh.)')
        print('Re-run with -o to bake it in and keep the URDF origin at zero instead.')
        return

    volume_ratio = out.signed_volume() / src.signed_volume() if src.signed_volume() else 0.0
    out.save(args.output)
    check = Stl.load(args.output)

    print(f'\nwrote {args.output}')
    print(f'  triangles          : {check.count} (source {src.count})')
    print(f'  header preserved   : {check.header == src.header}')
    print(f'  attr bytes kept    : {np.array_equal(check.attrs, src.attrs)}')
    print(f'  flipped faces      : {check.flipped_faces()} (source {src.flipped_faces()})')
    print(f'  signed volume ratio: {volume_ratio:.6f}  (must be +1; -1 means mirrored)')
    print('\nThe URDF <origin> for this mesh can now be all zeros.')

    if abs(volume_ratio - 1.0) > 1e-4 or check.count != src.count:
        raise SystemExit('verification FAILED - do not use this output')


if __name__ == '__main__':
    main()
