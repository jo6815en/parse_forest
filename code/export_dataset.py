import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a transformed COLMAP reconstruction."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder under datasets/, for example: torpa",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Reconstruction run, for example: 20260824_142747",
    )

    return parser.parse_args()


def qvec2rotmat(q):
    """
    COLMAP quaternion convention:
    q = [qw, qx, qy, qz]

    Returns the world-to-camera / world-to-rig rotation.
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


def camera_to_world(q, t):
    """
    Convert COLMAP world-to-rig/world-to-camera pose to
    rig/camera-to-world.

    COLMAP:
        x_rig = R * x_world + t

    Therefore:
        R_world = R.T
        C_world = -R.T @ t
    """
    R = qvec2rotmat(q)
    t = np.asarray(t, dtype=float)

    T_camera_to_world = np.eye(4)
    T_camera_to_world[:3, :3] = R.T
    T_camera_to_world[:3, 3] = -R.T @ t

    return T_camera_to_world


def frame_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def read_images(images_txt):
    """
    Read images.txt and return metadata indexed by IMAGE_ID.

    COLMAP images.txt contains two lines per image:
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D...
    """
    images = {}

    with open(images_txt, "r") as f:
        lines = f.readlines()

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

        if len(parts) < 10:
            i += 1
            continue

        camera_id = int(parts[8])
        name = parts[9]

        images[image_id] = {
            "camera_id": camera_id,
            "name": name,
        }

        # Skip the POINTS2D line.
        i += 2

    return images


def read_frames(frames_txt):
    """
    Read frames.txt.

    COLMAP format:
        FRAME_ID RIG_ID
        QW QX QY QZ TX TY TZ
        NUM_DATA_IDS
        SENSOR_TYPE SENSOR_ID DATA_ID ...

    The pose is RIG_FROM_WORLD.

    Returns:
        image_id -> frame pose
    """
    poses = {}

    with open(frames_txt, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 10:
            continue

        try:
            int(parts[0])  # FRAME_ID
            int(parts[1])  # RIG_ID
        except ValueError:
            continue

        q = list(map(float, parts[2:6]))
        t = list(map(float, parts[6:9]))

        num_data_ids = int(parts[9])

        T_camera_to_world = camera_to_world(q, t)

        offset = 10

        for _ in range(num_data_ids):
            if offset + 2 >= len(parts):
                break

            sensor_type = parts[offset]
            sensor_id = parts[offset + 1]
            data_id = int(parts[offset + 2])

            # For the current dataset, each rig contains one camera
            # and each frame contains one image.
            poses[data_id] = {
                "sensor_type": sensor_type,
                "sensor_id": sensor_id,
                "T_camera_to_world": T_camera_to_world,
            }

            offset += 3

    return poses


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    dataset_root = (
        repo_root
        / "datasets"
        / args.dataset
    )

    run_root = (
        dataset_root
        / "reconstructions"
        / args.run
    )

    colmap_dir = run_root / "colmap"

    # --------------------------------------------------
    # IMPORTANT:
    # Read the Y-up transformed model.
    # --------------------------------------------------

    model_dir = colmap_dir / "sparse_yup_txt"

    images_txt = model_dir / "images.txt"
    frames_txt = model_dir / "frames.txt"

    image_dir = dataset_root / "images"

    out_dir = run_root / "reconstruction"
    out_images_dir = out_dir / "images"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find transformed COLMAP model:\n{model_dir}"
        )

    if not images_txt.exists():
        raise FileNotFoundError(
            f"Could not find:\n{images_txt}"
        )

    if not frames_txt.exists():
        raise FileNotFoundError(
            f"Could not find:\n{frames_txt}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Could not find image directory:\n{image_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Read image metadata
    # --------------------------------------------------

    images = read_images(images_txt)

    # --------------------------------------------------
    # Read transformed frame poses
    # --------------------------------------------------

    frame_poses = read_frames(frames_txt)

    # --------------------------------------------------
    # Connect FRAME DATA_ID -> IMAGE_ID -> image name
    # --------------------------------------------------

    poses = {}

    for image_id, frame_data in frame_poses.items():

        if image_id not in images:
            continue

        name = images[image_id]["name"]

        poses[name] = {
            "image_id": image_id,
            "camera_id": images[image_id]["camera_id"],
            "T_camera_to_world": frame_data["T_camera_to_world"],
        }

    if not poses:
        raise RuntimeError(
            "No camera poses could be connected between "
            "frames.txt and images.txt."
        )

    names = sorted(
        poses.keys(),
        key=frame_number,
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    metadata = {
        "pose_convention": (
            "T_camera_to_world maps camera coordinates "
            "to Y-up world coordinates."
        ),
        "world_up_axis": "Y",
        "scale": (
            "COLMAP scale is arbitrary unless externally calibrated."
        ),
        "num_frames": len(names),
        "source_model": str(model_dir),
        "frames": [],
    }

    camera_poses = []

    # --------------------------------------------------
    # Export images + camera poses
    # --------------------------------------------------

    for idx, name in enumerate(names):

        new_name = f"{idx:06d}.jpg"

        source_image = image_dir / name
        target_image = out_images_dir / new_name

        if not source_image.exists():
            raise FileNotFoundError(
                f"Image referenced by COLMAP does not exist:\n"
                f"{source_image}"
            )

        shutil.copy(
            source_image,
            target_image,
        )

        T = poses[name]["T_camera_to_world"]

        R = T[:3, :3]
        position = T[:3, 3]

        frame_data = {
            "index": idx,
            "original_name": name,
            "dataset_name": new_name,
            "image_id": poses[name]["image_id"],
            "camera_id": poses[name]["camera_id"],
            "position_world": position.tolist(),
            "rotation_matrix": R.tolist(),
            "T_camera_to_world": T.tolist(),
        }

        metadata["frames"].append(frame_data)
        camera_poses.append(frame_data)

    # --------------------------------------------------
    # Relative poses
    # --------------------------------------------------

    relative = []

    for idx in range(len(names) - 1):

        a = names[idx]
        b = names[idx + 1]

        T_wa = poses[a]["T_camera_to_world"]
        T_wb = poses[b]["T_camera_to_world"]

        # Coordinates from camera a -> camera b.
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

    # --------------------------------------------------
    # Write files
    # --------------------------------------------------

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    with open(out_dir / "camera_poses.json", "w") as f:
        json.dump(
            camera_poses,
            f,
            indent=2,
        )

    with open(out_dir / "relative_poses.json", "w") as f:
        json.dump(
            relative,
            f,
            indent=2,
        )

    print()
    print("Export complete")
    print("----------------------------------------")
    print(f"Dataset:          {args.dataset}")
    print(f"Run:              {args.run}")
    print(f"Source model:     {model_dir}")
    print(f"Registered imgs:  {len(names)}")
    print(f"Relative poses:   {len(relative)}")
    print(f"World up axis:    Y")
    print(f"Output:           {out_dir}")
    print()


if __name__ == "__main__":
    main()