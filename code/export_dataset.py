import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def camera_to_world(q, t):
    R = qvec2rotmat(q)
    t = np.asarray(t, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def frame_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def read_images(images_txt):
    images = {}
    lines = images_txt.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        try:
            image_id = int(parts[0])
        except (ValueError, IndexError):
            i += 1
            continue
        if len(parts) >= 10:
            images[image_id] = {
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        i += 2
    return images


def read_frames(frames_txt):
    poses = {}
    for raw in frames_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0]); int(parts[1])
        except ValueError:
            continue
        q = list(map(float, parts[2:6]))
        t = list(map(float, parts[6:9]))
        num_data_ids = int(parts[9])
        T = camera_to_world(q, t)
        offset = 10
        for _ in range(num_data_ids):
            if offset + 2 >= len(parts):
                break
            data_id = int(parts[offset + 2])
            poses[data_id] = T
            offset += 3
    return poses


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "datasets" / args.dataset
    run_root = dataset_root / "reconstructions" / args.run
    model_dir = run_root / "colmap" / "sparse_yup_txt"
    image_dir = run_root / "images"
    out_dir = run_root / "reconstruction"
    out_images = out_dir / "images"

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing Y-up sparse model: {model_dir}")
    if not (model_dir / "images.txt").exists():
        raise FileNotFoundError(f"Missing {model_dir / 'images.txt'}")
    if not (model_dir / "frames.txt").exists():
        raise FileNotFoundError(f"Missing {model_dir / 'frames.txt'}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing run images: {image_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    images = read_images(model_dir / "images.txt")
    frame_poses = read_frames(model_dir / "frames.txt")
    poses = {}
    for image_id, T in frame_poses.items():
        if image_id in images:
            poses[images[image_id]["name"]] = {
                "image_id": image_id,
                "camera_id": images[image_id]["camera_id"],
                "T_camera_to_world": T,
            }

    if not poses:
        raise RuntimeError("Could not connect frames.txt to images.txt")

    names = sorted(poses, key=frame_number)
    metadata = {
        "pose_convention": "T_camera_to_world maps camera coordinates to Y-up world coordinates.",
        "world_up_axis": "Y",
        "scale": "COLMAP scale is arbitrary unless externally calibrated.",
        "num_frames": len(names),
        "frames": [],
    }
    camera_poses = []

    for idx, name in enumerate(names):
        dataset_name = f"{idx:06d}.jpg"
        src = image_dir / name
        dst = out_images / dataset_name
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        shutil.copy(src, dst)
        T = poses[name]["T_camera_to_world"]
        frame = {
            "index": idx,
            "original_name": name,
            "dataset_name": dataset_name,
            "image_id": poses[name]["image_id"],
            "camera_id": poses[name]["camera_id"],
            "position_world": T[:3, 3].tolist(),
            "rotation_matrix": T[:3, :3].tolist(),
            "T_camera_to_world": T.tolist(),
        }
        metadata["frames"].append(frame)
        camera_poses.append(frame)

    relative = []
    for idx in range(len(names) - 1):
        T_a = poses[names[idx]]["T_camera_to_world"]
        T_b = poses[names[idx + 1]]["T_camera_to_world"]
        relative.append({
            "from_index": idx,
            "to_index": idx + 1,
            "from_image": f"{idx:06d}.jpg",
            "to_image": f"{idx + 1:06d}.jpg",
            "original_from": names[idx],
            "original_to": names[idx + 1],
            "T_a_to_b": (np.linalg.inv(T_b) @ T_a).tolist(),
        })

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "camera_poses.json").write_text(json.dumps(camera_poses, indent=2))
    (out_dir / "relative_poses.json").write_text(json.dumps(relative, indent=2))
    print(f"Exported {len(camera_poses)} camera poses to {out_dir}")


if __name__ == "__main__":
    main()
