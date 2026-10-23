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


def rigid_transform(q, t):
    """Return the 4x4 transform represented by COLMAP quaternion q and t."""
    T = np.eye(4)
    T[:3, :3] = qvec2rotmat(q)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def frame_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def read_cameras(cameras_txt):
    """Read intrinsics for the undistorted COLMAP images."""
    cameras = {}

    for raw in cameras_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))

        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            f, cx, cy = params
            fx = fy = f
        else:
            raise ValueError(
                f"Unsupported dense camera model {model!r} for camera "
                f"{camera_id}. Expected PINHOLE or SIMPLE_PINHOLE."
            )

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ])

        cameras[camera_id] = {
            "camera_id": camera_id,
            "camera_model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": K.tolist(),
        }

    if not cameras:
        raise RuntimeError(f"No cameras found in {cameras_txt}")

    return cameras


def read_images(images_txt):
    """Return COLMAP image records keyed by image id and by image name."""
    by_id = {}
    lines = images_txt.read_text().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        try:
            image_id = int(parts[0])
            camera_id = int(parts[8])
        except ValueError:
            i += 1
            continue

        by_id[image_id] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "name": parts[9],
        }

        # A COLMAP image record occupies two lines. The second line contains
        # the 2D observations and may be empty.
        i += 2

    if not by_id:
        raise RuntimeError(f"No images found in {images_txt}")

    by_name = {record["name"]: record for record in by_id.values()}
    return by_id, by_name


def read_rigs(rigs_txt):
    """Read SENSOR_FROM_RIG transforms, including identity reference sensors."""
    sensor_from_rig = {}

    for raw in rigs_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        rig_id = int(parts[0])
        num_sensors = int(parts[1])
        ref_sensor_type = parts[2]
        ref_sensor_id = int(parts[3])

        sensor_from_rig[(rig_id, ref_sensor_type, ref_sensor_id)] = np.eye(4)

        offset = 4
        for _ in range(num_sensors - 1):
            sensor_type = parts[offset]
            sensor_id = int(parts[offset + 1])
            has_pose = int(parts[offset + 2])
            offset += 3

            if has_pose:
                q = list(map(float, parts[offset:offset + 4]))
                t = list(map(float, parts[offset + 4:offset + 7]))
                T_sensor_from_rig = rigid_transform(q, t)
                offset += 7
            else:
                T_sensor_from_rig = np.eye(4)

            sensor_from_rig[(rig_id, sensor_type, sensor_id)] = T_sensor_from_rig

    if not sensor_from_rig:
        raise RuntimeError(f"No rigs found in {rigs_txt}")

    return sensor_from_rig


def read_camera_poses(frames_txt, sensor_from_rig):
    """Return Y-up CAMERA_TO_WORLD transforms keyed by COLMAP image id."""
    poses = {}

    for raw in frames_txt.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        frame_id = int(parts[0])
        rig_id = int(parts[1])
        q_rig_from_world = list(map(float, parts[2:6]))
        t_rig_from_world = list(map(float, parts[6:9]))
        num_data_ids = int(parts[9])
        T_rig_from_world = rigid_transform(q_rig_from_world, t_rig_from_world)

        offset = 10
        for _ in range(num_data_ids):
            sensor_type = parts[offset]
            sensor_id = int(parts[offset + 1])
            data_id = int(parts[offset + 2])
            offset += 3

            if sensor_type != "CAMERA":
                continue

            key = (rig_id, sensor_type, sensor_id)
            if key not in sensor_from_rig:
                raise KeyError(
                    f"Frame {frame_id} refers to missing rig sensor {key}"
                )

            T_camera_from_world = sensor_from_rig[key] @ T_rig_from_world
            poses[data_id] = np.linalg.inv(T_camera_from_world)

    if not poses:
        raise RuntimeError(f"No camera poses found in {frames_txt}")

    return poses


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "datasets" / args.dataset / "reconstructions" / args.run

    y_up_model = run_root / "colmap" / "sparse_yup_txt"
    dense_model = run_root / "colmap" / "dense" / "sparse_txt"
    original_image_dir = run_root / "images"
    undistorted_image_dir = run_root / "colmap" / "dense" / "images"

    out_dir = run_root / "reconstruction"
    out_original_images = out_dir / "images"
    out_undistorted_images = out_dir / "images_undistorted"

    required_files = [
        y_up_model / "images.txt",
        y_up_model / "frames.txt",
        y_up_model / "rigs.txt",
        dense_model / "cameras.txt",
        dense_model / "images.txt",
    ]
    for path in required_files:
        require_file(path)

    if not original_image_dir.is_dir():
        raise FileNotFoundError(f"Missing original image directory: {original_image_dir}")
    if not undistorted_image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing undistorted image directory: {undistorted_image_dir}"
        )

    y_up_images_by_id, _ = read_images(y_up_model / "images.txt")
    _, dense_images_by_name = read_images(dense_model / "images.txt")
    dense_cameras = read_cameras(dense_model / "cameras.txt")
    sensor_from_rig = read_rigs(y_up_model / "rigs.txt")
    poses_by_image_id = read_camera_poses(
        y_up_model / "frames.txt",
        sensor_from_rig,
    )

    pose_by_name = {}
    for image_id, T_camera_to_world in poses_by_image_id.items():
        if image_id not in y_up_images_by_id:
            raise KeyError(
                f"Y-up frame refers to image id {image_id}, but that id is "
                f"missing from {y_up_model / 'images.txt'}"
            )
        name = y_up_images_by_id[image_id]["name"]
        pose_by_name[name] = {
            "image_id": image_id,
            "T_camera_to_world": T_camera_to_world,
        }

    missing_dense_records = sorted(set(pose_by_name) - set(dense_images_by_name))
    if missing_dense_records:
        raise RuntimeError(
            "Registered Y-up images are missing from the dense model: "
            + ", ".join(missing_dense_records[:10])
        )

    names = sorted(pose_by_name, key=frame_number)
    if not names:
        raise RuntimeError("No common registered images were found")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_original_images.mkdir(parents=True, exist_ok=True)
    out_undistorted_images.mkdir(parents=True, exist_ok=True)

    camera_poses = []

    for index, name in enumerate(names):
        dataset_name = f"{index:06d}.jpg"
        original_src = original_image_dir / name
        undistorted_src = undistorted_image_dir / name

        if not original_src.is_file():
            raise FileNotFoundError(f"Missing original image: {original_src}")
        if not undistorted_src.is_file():
            raise FileNotFoundError(f"Missing undistorted image: {undistorted_src}")

        shutil.copy2(original_src, out_original_images / dataset_name)
        shutil.copy2(undistorted_src, out_undistorted_images / dataset_name)

        dense_image = dense_images_by_name[name]
        camera_id = dense_image["camera_id"]
        if camera_id not in dense_cameras:
            raise KeyError(f"Missing dense camera id {camera_id} for image {name}")

        intrinsics = dense_cameras[camera_id]
        T_camera_to_world = pose_by_name[name]["T_camera_to_world"]
        T_world_to_camera = np.linalg.inv(T_camera_to_world)

        frame = {
            "index": index,
            "original_name": name,
            "dataset_name": dataset_name,
            "original_image_path": f"images/{dataset_name}",
            "undistorted_image_path": f"images_undistorted/{dataset_name}",
            "image_id": pose_by_name[name]["image_id"],
            **intrinsics,
            "distortion": [],
            "position_world": T_camera_to_world[:3, 3].tolist(),
            "rotation_matrix": T_camera_to_world[:3, :3].tolist(),
            "R_camera_to_world": T_camera_to_world[:3, :3].tolist(),
            "R_world_to_camera": T_world_to_camera[:3, :3].tolist(),
            "T_camera_to_world": T_camera_to_world.tolist(),
            "T_world_to_camera": T_world_to_camera.tolist(),
        }
        camera_poses.append(frame)

    relative_poses = []
    for frame_a, frame_b in zip(camera_poses[:-1], camera_poses[1:]):
        T_a = np.asarray(frame_a["T_camera_to_world"])
        T_b = np.asarray(frame_b["T_camera_to_world"])
        relative_poses.append({
            "from_index": frame_a["index"],
            "to_index": frame_b["index"],
            "from_image": frame_a["dataset_name"],
            "to_image": frame_b["dataset_name"],
            "original_from": frame_a["original_name"],
            "original_to": frame_b["original_name"],
            "T_a_to_b": (np.linalg.inv(T_b) @ T_a).tolist(),
        })

    metadata = {
        "pose_convention": (
            "T_camera_to_world maps camera coordinates to Y-up world coordinates."
        ),
        "intrinsics_convention": (
            "K is for the corresponding image in images_undistorted/. "
            "Those images have already been undistorted by COLMAP."
        ),
        "world_up_axis": "Y",
        "scale": "COLMAP scale is arbitrary unless externally calibrated.",
        "num_frames": len(camera_poses),
        "frames": camera_poses,
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (out_dir / "camera_poses.json").write_text(
        json.dumps(camera_poses, indent=2),
        encoding="utf-8",
    )
    (out_dir / "relative_poses.json").write_text(
        json.dumps(relative_poses, indent=2),
        encoding="utf-8",
    )

    print(f"Exported {len(camera_poses)} frames to: {out_dir}")
    print(f"Original images:    {out_original_images}")
    print(f"Undistorted images: {out_undistorted_images}")
    print(f"Camera data:        {out_dir / 'camera_poses.json'}")


if __name__ == "__main__":
    main()
