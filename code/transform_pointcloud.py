import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform a COLMAP dense point cloud using a 4x4 world transformation."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example: torpa",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Reconstruction run, for example: 20260824_142747",
    )

    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to the 4x4 transformation matrix.",
    )

    return parser.parse_args()


def load_matrix(path):
    matrix = np.loadtxt(path)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 matrix, got {matrix.shape}"
        )

    return matrix


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    run_root = (
        repo_root
        / "datasets"
        / args.dataset
        / "reconstructions"
        / args.run
        / "colmap"
    )

    input_file = run_root / "dense" / "fused.ply"
    output_file = run_root / "dense" / "fused_yup.ply"

    matrix_path = Path(args.matrix)

    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find point cloud:\n{input_file}"
        )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Could not find transformation matrix:\n{matrix_path}"
        )

    T = load_matrix(matrix_path)

    R = T[:3, :3]
    t = T[:3, 3]

    print("\nTransformation:")
    print(T)

    print(f"\nReading:\n{input_file}")

    ply = PlyData.read(input_file)

    vertex = ply["vertex"].data

    # --------------------------------------------------
    # Transform XYZ
    # --------------------------------------------------

    xyz = np.column_stack([
        vertex["x"],
        vertex["y"],
        vertex["z"],
    ]).astype(np.float64)

    xyz_new = (R @ xyz.T).T + t

    vertex["x"] = xyz_new[:, 0]
    vertex["y"] = xyz_new[:, 1]
    vertex["z"] = xyz_new[:, 2]

    # --------------------------------------------------
    # Transform normals, if present
    # --------------------------------------------------

    names = vertex.dtype.names

    if all(name in names for name in ("nx", "ny", "nz")):

        normals = np.column_stack([
            vertex["nx"],
            vertex["ny"],
            vertex["nz"],
        ]).astype(np.float64)

        normals_new = (R @ normals.T).T

        # Normalize again to avoid numerical drift.
        lengths = np.linalg.norm(
            normals_new,
            axis=1,
            keepdims=True,
        )

        valid = lengths[:, 0] > 0

        normals_new[valid] /= lengths[valid]

        vertex["nx"] = normals_new[:, 0]
        vertex["ny"] = normals_new[:, 1]
        vertex["nz"] = normals_new[:, 2]

        print("Transformed normals.")

    # --------------------------------------------------
    # Write new PLY
    # --------------------------------------------------

    new_vertex = PlyElement.describe(
        vertex,
        "vertex",
    )

    output_ply = PlyData(
        [new_vertex],
        text=ply.text,
        byte_order=ply.byte_order,
    )

    output_ply.write(output_file)

    print("\nDone.")
    print(f"Points: {len(vertex)}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()