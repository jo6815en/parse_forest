import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a COLMAP dataset into a standardized reconstruction format."
    )
    parser.add_argument(
        "--dataset",
        default="forest_colmap",
        help="Dataset folder name under datasets/ (for example: forest_colmap or campus)",
    )
    return parser.parse_args()


args = parse_args()
DATASET_NAME = args.dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / DATASET_NAME
COLMAP_DIR = DATASET_ROOT / "colmap"

IMAGES_TXT = COLMAP_DIR / "sparse_txt" / "images.txt"
IMAGE_DIR = DATASET_ROOT / "images"

OUT_DIR = DATASET_ROOT / "reconstruction"


def qvec2rotmat(q):
    """
    COLMAP quaternion convention:
    q = [qw, qx, qy, qz]

    Returns the COLMAP world-to-camera rotation matrix R.
    """
    qw, qx, qy, qz = q

    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ])


def frame_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def camera_to_world(q, t):
    """
    Convert COLMAP's world-to-camera pose into camera-to-world.

    COLMAP:
        x_camera = R * x_world + t

    Therefore:
        R_cw = R^T
        C_world = -R^T * t
    """
    R = qvec2rotmat(q)
    t = np.asarray(t, dtype=float)

    T_camera_to_world = np.eye(4)
    T_camera_to_world[:3, :3] = R.T
    T_camera_to_world[:3, 3] = -R.T @ t

    return T_camera_to_world


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion [qw, qx, qy, qz].
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)

    return q


def main():
    poses = {}

    if not IMAGES_TXT.exists():
        raise FileNotFoundError(
            f"COLMAP images.txt not found: {IMAGES_TXT}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    with open(IMAGES_TXT, "r") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#") or line == "":
            i += 1
            continue

        parts = line.split()

        try:
            int(parts[0])
        except ValueError:
            i += 1
            continue

        # COLMAP images.txt:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        q = list(map(float, parts[1:5]))
        t = list(map(float, parts[5:8]))
        camera_id = int(parts[8])
        name = parts[9]

        T_camera_to_world = camera_to_world(q, t)

        poses[name] = {
            "camera_id": camera_id,
            "q_colmap": q,
            "t_colmap": t,
            "T_camera_to_world": T_camera_to_world,
        }

        # Skip the following 2D-point line.
        i += 2

    names = sorted(poses.keys(), key=frame_number)

    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    metadata = {
        "pose_convention": (
            "T_camera_to_world maps coordinates from camera coordinates "
            "to COLMAP world coordinates."
        ),
        "scale": (
            "COLMAP scale is arbitrary unless externally calibrated."
        ),
        "num_frames": len(names),
        "frames": [],
    }

    camera_poses = []

    for idx, name in enumerate(names):
        new_name = f"{idx:06d}.jpg"

        source_image = IMAGE_DIR / name
        target_image = OUT_DIR / "images" / new_name

        if not source_image.exists():
            raise FileNotFoundError(
                f"Image referenced by COLMAP does not exist: {source_image}"
            )

        shutil.copy(source_image, target_image)

        T = poses[name]["T_camera_to_world"]
        R = T[:3, :3]
        position = T[:3, 3]
        quaternion = rotation_matrix_to_quaternion(R)

        frame_data = {
            "index": idx,
            "original_name": name,
            "dataset_name": new_name,
            "camera_id": poses[name]["camera_id"],
            "position_world": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "rotation_matrix": R.tolist(),
            "T_camera_to_world": T.tolist(),
        }

        metadata["frames"].append(frame_data)

        camera_poses.append(frame_data)

    relative = []

    for idx in range(len(names) - 1):
        a = names[idx]
        b = names[idx + 1]

        T_wa = poses[a]["T_camera_to_world"]
        T_wb = poses[b]["T_camera_to_world"]

        # Transform coordinates from camera a to camera b.
        T_a_to_b = np.linalg.inv(T_wb) @ T_wa

        relative.append({
            "from_index": idx,
            "to_index": idx + 1,
            "from_image": f"{idx:06d}.jpg",
            "to_image": f"{idx + 1:06d}.jpg",
            "original_from": a,
            "original_to": b,
            "T_a_to_b": T_a_to_b.tolist(),
        })

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(OUT_DIR / "camera_poses.json", "w") as f:
        json.dump(camera_poses, f, indent=2)

    with open(OUT_DIR / "relative_poses.json", "w") as f:
        json.dump(relative, f, indent=2)

    print(f"Saved dataset with {len(names)} images")
    print(f"Saved {len(names)} camera poses")
    print(f"Saved {len(relative)} relative poses")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()