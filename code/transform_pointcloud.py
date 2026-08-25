import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--matrix", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "datasets" / args.dataset / "reconstructions" / args.run / "colmap"
    input_file = run_root / "dense" / "fused.ply"
    output_file = run_root / "dense" / "fused_yup.ply"
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find point cloud: {input_file}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Could not find matrix: {matrix_path}")

    T = np.loadtxt(matrix_path)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]

    ply = PlyData.read(input_file)
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    xyz_new = (R @ xyz.T).T + t
    vertex["x"] = xyz_new[:, 0]
    vertex["y"] = xyz_new[:, 1]
    vertex["z"] = xyz_new[:, 2]

    names = vertex.dtype.names
    if all(name in names for name in ("nx", "ny", "nz")):
        normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(np.float64)
        normals_new = (R @ normals.T).T
        lengths = np.linalg.norm(normals_new, axis=1, keepdims=True)
        valid = lengths[:, 0] > 0
        normals_new[valid] /= lengths[valid]
        vertex["nx"] = normals_new[:, 0]
        vertex["ny"] = normals_new[:, 1]
        vertex["nz"] = normals_new[:, 2]

    new_vertex = PlyElement.describe(vertex, "vertex")
    PlyData([new_vertex], text=ply.text, byte_order=ply.byte_order).write(output_file)
    print(f"Wrote {output_file} ({len(vertex)} points)")


if __name__ == "__main__":
    main()
