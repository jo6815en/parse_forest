import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Level COLMAP images and cameras in a Y-up world using the "
            "per-image intrinsics exported by export_dataset.py."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N frames (useful for a visual test).",
    )
    return parser.parse_args()


def level_camera_rotation(R_camera_to_world):
    """
    Return an upright CAMERA_TO_WORLD rotation.

    COLMAP camera coordinates are x right, y down, z forward. The rectified
    camera therefore gets y-down = -world-up. Its forward direction is the
    original forward direction projected onto the horizontal XZ plane. This
    removes pitch and roll while preserving heading.
    """
    R_camera_to_world = np.asarray(R_camera_to_world, dtype=float)
    if R_camera_to_world.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 camera rotation, got {R_camera_to_world.shape}"
        )

    world_up = np.array([0.0, 1.0, 0.0])
    old_forward = R_camera_to_world[:, 2]

    new_forward = old_forward - np.dot(old_forward, world_up) * world_up
    forward_norm = np.linalg.norm(new_forward)
    if forward_norm < 1e-8:
        raise ValueError(
            "Cannot level a camera whose forward axis is parallel to world up"
        )
    new_forward /= forward_norm

    new_down = -world_up
    new_right = np.cross(new_down, new_forward)
    new_right /= np.linalg.norm(new_right)

    # Recompute forward to remove accumulated numerical error.
    new_forward = np.cross(new_right, new_down)
    new_forward /= np.linalg.norm(new_forward)

    R_rectified_camera_to_world = np.column_stack([
        new_right,
        new_down,
        new_forward,
    ])

    if not np.allclose(
        R_rectified_camera_to_world.T @ R_rectified_camera_to_world,
        np.eye(3),
        atol=1e-6,
    ):
        raise RuntimeError("Computed rectified rotation is not orthonormal")

    if not np.isclose(
        np.linalg.det(R_rectified_camera_to_world),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError("Computed rectified rotation is not a proper rotation")

    return R_rectified_camera_to_world


def rotation_homography(K, R_source_camera_to_world, R_target_camera_to_world):
    """
    Map pixels from the source camera to the target camera.

    Both cameras have the same center and intrinsics; only their orientation
    differs.
    """
    K = np.asarray(K, dtype=float)
    R_source = np.asarray(R_source_camera_to_world, dtype=float)
    R_target = np.asarray(R_target_camera_to_world, dtype=float)

    if K.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 K matrix, got {K.shape}")

    H = K @ R_target.T @ R_source @ np.linalg.inv(K)

    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]

    return H


def make_pose(R_camera_to_world, position_world):
    T_camera_to_world = np.eye(4)
    T_camera_to_world[:3, :3] = R_camera_to_world
    T_camera_to_world[:3, 3] = position_world
    T_world_to_camera = np.linalg.inv(T_camera_to_world)
    return T_camera_to_world, T_world_to_camera


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "datasets" / args.dataset / "reconstructions" / args.run
    reconstruction_dir = run_root / "reconstruction"
    poses_file = reconstruction_dir / "camera_poses.json"

    output_dir = reconstruction_dir / "rectified"
    output_images = output_dir / "images"
    output_homographies = output_dir / "homographies"

    if not poses_file.is_file():
        raise FileNotFoundError(
            f"Missing {poses_file}. Run export_dataset.py first."
        )

    frames = json.loads(poses_file.read_text(encoding="utf-8"))
    if not frames:
        raise RuntimeError(f"No camera frames found in {poses_file}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        frames = frames[:args.limit]

    output_images.mkdir(parents=True, exist_ok=True)
    output_homographies.mkdir(parents=True, exist_ok=True)

    rectified_frames = []

    for number, frame in enumerate(frames, start=1):
        required = [
            "dataset_name",
            "undistorted_image_path",
            "width",
            "height",
            "K",
            "R_camera_to_world",
            "position_world",
        ]
        missing = [key for key in required if key not in frame]
        if missing:
            raise KeyError(
                f"Frame {frame.get('index', number - 1)} is missing: "
                + ", ".join(missing)
            )

        source_image = reconstruction_dir / frame["undistorted_image_path"]
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing undistorted image: {source_image}")

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read: {source_image}")

        height, width = image.shape[:2]
        expected_width = int(frame["width"])
        expected_height = int(frame["height"])
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"Image {source_image.name} is {width}x{height}, but its "
                f"camera model says {expected_width}x{expected_height}"
            )

        K = np.asarray(frame["K"], dtype=float)
        R_source = np.asarray(frame["R_camera_to_world"], dtype=float)
        position_world = np.asarray(frame["position_world"], dtype=float)

        R_rectified = level_camera_rotation(R_source)
        H = rotation_homography(K, R_source, R_rectified)

        rectified_image = cv2.warpPerspective(
            image,
            H,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        dataset_name = frame["dataset_name"]
        output_image = output_images / dataset_name
        if not cv2.imwrite(
            str(output_image),
            rectified_image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"Could not write rectified image: {output_image}")

        homography_name = f"{Path(dataset_name).stem}.txt"
        np.savetxt(output_homographies / homography_name, H, fmt="%.12f")

        T_camera_to_world, T_world_to_camera = make_pose(
            R_rectified,
            position_world,
        )

        rectified_frame = dict(frame)
        rectified_frame.update({
            "source_undistorted_image_path": (
                f"../{frame['undistorted_image_path']}"
            ),
            "image_path": f"images/{dataset_name}",
            "rectified_image_path": f"images/{dataset_name}",
            "homography_path": f"homographies/{homography_name}",
            "H_undistorted_to_rectified": H.tolist(),
            "position_world": position_world.tolist(),
            "rotation_matrix": R_rectified.tolist(),
            "R_camera_to_world": R_rectified.tolist(),
            "R_world_to_camera": R_rectified.T.tolist(),
            "T_camera_to_world": T_camera_to_world.tolist(),
            "T_world_to_camera": T_world_to_camera.tolist(),
        })
        rectified_frames.append(rectified_frame)

        print(f"[{number}/{len(frames)}] {dataset_name}")

    relative_poses = []
    for frame_a, frame_b in zip(rectified_frames[:-1], rectified_frames[1:]):
        T_a = np.asarray(frame_a["T_camera_to_world"], dtype=float)
        T_b = np.asarray(frame_b["T_camera_to_world"], dtype=float)
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
        "description": (
            "Images and cameras leveled in a Y-up world. Camera y-down is "
            "aligned with -world-Y, and camera forward is horizontal."
        ),
        "pose_convention": (
            "T_camera_to_world maps rectified camera coordinates to "
            "Y-up world coordinates."
        ),
        "image_source": (
            "Each image was warped from its COLMAP-undistorted source image."
        ),
        "world_up_axis": "Y",
        "num_frames": len(rectified_frames),
        "frames": rectified_frames,
    }

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "camera_poses.json", rectified_frames)
    write_json(output_dir / "relative_poses.json", relative_poses)

    print()
    print(f"Rectified {len(rectified_frames)} images.")
    print(f"Output: {output_dir}")
    print(f"Images: {output_images}")
    print(f"Camera poses: {output_dir / 'camera_poses.json'}")
    print(f"Relative poses: {output_dir / 'relative_poses.json'}")


if __name__ == "__main__":
    main()
