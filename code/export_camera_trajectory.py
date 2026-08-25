import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export camera trajectory as a PLY file."
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

    return parser.parse_args()


def write_ply(path, points, lines):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")

        f.write("end_header\n")

        for point in points:
            f.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f}\n"
            )

        for a, b in lines:
            f.write(f"{a} {b}\n")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    reconstruction_dir = (
        repo_root
        / "datasets"
        / args.dataset
        / "reconstructions"
        / args.run
        / "reconstruction"
    )

    poses_file = reconstruction_dir / "camera_poses.json"
    output_file = reconstruction_dir / "camera_trajectory.ply"

    if not poses_file.exists():
        raise FileNotFoundError(
            f"Could not find camera poses:\n{poses_file}"
        )

    with open(poses_file, "r") as f:
        poses = json.load(f)

    if not poses:
        raise RuntimeError(
            "camera_poses.json contains no poses."
        )

    points = np.asarray(
        [
            pose["position_world"]
            for pose in poses
        ],
        dtype=float,
    )

    lines = [
        (i, i + 1)
        for i in range(len(points) - 1)
    ]

    write_ply(
        output_file,
        points,
        lines,
    )

    print(f"Camera positions: {len(points)}")
    print(f"Trajectory segments: {len(lines)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()