import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    reconstruction = repo_root / "datasets" / args.dataset / "reconstructions" / args.run / "reconstruction"
    poses_file = reconstruction / "camera_poses.json"
    output_file = reconstruction / "camera_trajectory.ply"

    if not poses_file.exists():
        raise FileNotFoundError(f"Missing {poses_file}")

    poses = json.loads(poses_file.read_text())
    points = [p["position_world"] for p in poses]
    lines = [(i, i + 1) for i in range(len(points) - 1)]

    with open(output_file, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for a, b in lines:
            f.write(f"{a} {b}\n")

    print(f"Wrote {output_file} with {len(points)} camera positions")


if __name__ == "__main__":
    main()
