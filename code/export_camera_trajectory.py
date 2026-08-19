import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export COLMAP camera trajectory as a PLY point cloud."
    )
    parser.add_argument(
        "--dataset",
        default="forest_colmap",
        help="Dataset folder name under datasets/ (for example: campus)",
    )
    return parser.parse_args()


def write_ply(path, points, lines):
    """
    Write an ASCII PLY containing points and line segments.
    """
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
    dataset_root = repo_root / "datasets" / args.dataset
    reconstruction_dir = dataset_root / "reconstruction"

    poses_file = reconstruction_dir / "camera_poses.json"
    output_file = reconstruction_dir / "camera_trajectory.ply"

    if not poses_file.exists():
        raise FileNotFoundError(
            f"Could not find camera poses: {poses_file}"
        )

    with open(poses_file, "r") as f:
        poses = json.load(f)

    if not poses:
        raise RuntimeError("camera_poses.json contains no camera poses.")

    points = []

    for pose in poses:
        position = pose["position_world"]
        points.append(position)

    points = np.asarray(points, dtype=float)

    # One line between every consecutive camera position.
    lines = [
        (i, i + 1)
        for i in range(len(points) - 1)
    ]

    write_ply(output_file, points, lines)

    print(f"Saved {len(points)} camera positions")
    print(f"Saved {len(lines)} trajectory segments")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()