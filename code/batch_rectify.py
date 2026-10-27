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
        "--start",
        type=int,
        default=0,
        help="Start at this exported frame index. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N frames (useful for a visual test).",
    )
    parser.add_argument(
        "--fit-margin",
        type=float,
        default=0.98,
        help=(
            "Margin used when a rotated image must be scaled down to fit. "
            "Default: 0.98."
        ),
    )
    parser.add_argument(
        "--max-trusted-tilt",
        type=float,
        default=15.0,
        help=(
            "Pose tilt angles above this value are treated as reconstruction "
            "outliers and interpolated from reliable frames. Use 0 to trust "
            "all poses. Default: 15 degrees."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help=(
            "Odd rolling-median window for local pitch/roll corrections. "
            "Use 1 to disable smoothing. Default: 7."
        ),
    )
    return parser.parse_args()


def level_camera_rotation(R_camera_to_world):
    """
    Return the upright CAMERA_TO_WORLD rotation requiring the smallest change.

    COLMAP camera coordinates are x right, y down, z forward, so camera up in
    world coordinates is -R[:, 1]. We apply the unique minimum-angle rotation
    that maps this vector to world +Y. Unlike constructing a new camera basis
    from a projected forward vector, this introduces no unnecessary yaw.
    """
    R_camera_to_world = np.asarray(R_camera_to_world, dtype=float)
    if R_camera_to_world.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 camera rotation, got {R_camera_to_world.shape}"
        )

    if not np.allclose(
        R_camera_to_world.T @ R_camera_to_world,
        np.eye(3),
        atol=1e-5,
    ) or not np.isclose(np.linalg.det(R_camera_to_world), 1.0, atol=1e-5):
        raise ValueError("Source camera rotation is not a proper rotation")

    world_up = np.array([0.0, 1.0, 0.0])
    camera_up = -R_camera_to_world[:, 1]
    camera_up /= np.linalg.norm(camera_up)

    cosine = float(np.clip(np.dot(camera_up, world_up), -1.0, 1.0))
    axis_unnormalized = np.cross(camera_up, world_up)
    sine = float(np.linalg.norm(axis_unnormalized))

    if sine < 1e-10:
        if cosine > 0.0:
            Q_world = np.eye(3)
        else:
            # The camera is upside down. A 180-degree rotation has no unique
            # axis; using its forward axis preserves the viewing direction.
            axis = R_camera_to_world[:, 2]
            axis /= np.linalg.norm(axis)
            Q_world = 2.0 * np.outer(axis, axis) - np.eye(3)
    else:
        axis = axis_unnormalized / sine
        angle = np.arctan2(sine, cosine)
        x, y, z = axis
        K_axis = np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ])
        Q_world = (
            np.eye(3)
            + np.sin(angle) * K_axis
            + (1.0 - np.cos(angle)) * (K_axis @ K_axis)
        )

    R_rectified_camera_to_world = Q_world @ R_camera_to_world

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

    rectified_up = -R_rectified_camera_to_world[:, 1]
    if not np.allclose(rectified_up, world_up, atol=1e-6):
        raise RuntimeError("Rectified camera is not aligned with world up")

    return R_rectified_camera_to_world


def rotation_angle_degrees(R_source, R_target):
    relative = np.asarray(R_target).T @ np.asarray(R_source)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_matrix_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    cosine = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))

    if angle < 1e-10:
        return np.zeros(3)

    if np.pi - angle < 1e-6:
        values, vectors = np.linalg.eig(R)
        index = int(np.argmin(np.abs(values - 1.0)))
        axis = np.real(vectors[:, index])
        axis /= np.linalg.norm(axis)
        return axis * angle

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def rotvec_to_rotation_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        return np.eye(3)

    x, y, z = rotvec / angle
    K_axis = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return (
        np.eye(3)
        + np.sin(angle) * K_axis
        + (1.0 - np.cos(angle)) * (K_axis @ K_axis)
    )


def rolling_median(values, window):
    if window < 1 or window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer")
    if window == 1:
        return values.copy()

    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([
        np.median(padded[i:i + window], axis=0)
        for i in range(len(values))
    ])


def regularized_leveling_corrections(
    frames,
    max_trusted_tilt_degrees,
    smooth_window,
):
    """
    Return raw and regularized pitch/roll corrections in camera coordinates.

    Yaw is never smoothed. Large pose tilts are treated as SfM outliers and
    interpolated from trustworthy neighboring frames. At either sequence end,
    the closest trustworthy correction is held constant.
    """
    raw = []

    for frame in frames:
        R_source = np.asarray(frame["R_camera_to_world"], dtype=float)
        R_leveled = level_camera_rotation(R_source)
        Q_local = R_source.T @ R_leveled
        raw.append(rotation_matrix_to_rotvec(Q_local))

    raw = np.asarray(raw)
    raw_angles = np.degrees(np.linalg.norm(raw, axis=1))

    if max_trusted_tilt_degrees <= 0.0:
        trusted = np.ones(len(raw), dtype=bool)
    else:
        trusted = raw_angles <= max_trusted_tilt_degrees

    if not np.any(trusted):
        raise ValueError(
            "No camera pose is below --max-trusted-tilt; increase the limit"
        )

    indices = np.arange(len(raw), dtype=float)
    trusted_indices = indices[trusted]
    regularized = np.empty_like(raw)

    for component in range(3):
        regularized[:, component] = np.interp(
            indices,
            trusted_indices,
            raw[trusted, component],
        )

    regularized = rolling_median(regularized, smooth_window)
    return raw, regularized, trusted


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


def fit_homography_to_canvas(H, K_target, width, height, margin):
    """
    Scale and center a rotated view so its four source corners fit.

    The returned 2D similarity is also applied to the target intrinsics. This
    preserves the projective camera model while avoiding unnecessary clipping.
    """
    if not 0.0 < margin <= 1.0:
        raise ValueError("--fit-margin must be greater than 0 and at most 1")

    corners = np.array([
        [0.0, 0.0, 1.0],
        [width - 1.0, 0.0, 1.0],
        [width - 1.0, height - 1.0, 1.0],
        [0.0, height - 1.0, 1.0],
    ])
    warped_h = (H @ corners.T).T
    denominators = warped_h[:, 2]

    if np.any(np.abs(denominators) < 1e-10):
        raise ValueError("A transformed image corner lies at infinity")

    # A homography has arbitrary global sign. Mixed signs, however, mean that
    # its horizon intersects the image and no finite canvas can contain it.
    if np.min(denominators) < 0.0 < np.max(denominators):
        raise ValueError(
            "The rectified camera horizon intersects the source image"
        )

    warped = warped_h[:, :2] / denominators[:, None]
    minimum = warped.min(axis=0)
    maximum = warped.max(axis=0)
    span = maximum - minimum

    if np.any(span <= 1e-10):
        raise ValueError("The transformed image has a degenerate extent")

    scale = min(
        (width - 1.0) / span[0],
        (height - 1.0) / span[1],
        1.0,
    )
    if scale < 1.0:
        scale *= margin

    source_center = 0.5 * (minimum + maximum)
    target_center = np.array([
        0.5 * (width - 1.0),
        0.5 * (height - 1.0),
    ])
    translation = target_center - scale * source_center

    canvas_transform = np.array([
        [scale, 0.0, translation[0]],
        [0.0, scale, translation[1]],
        [0.0, 0.0, 1.0],
    ])

    H_fitted = canvas_transform @ H
    K_fitted = canvas_transform @ K_target

    if abs(H_fitted[2, 2]) > 1e-12:
        H_fitted /= H_fitted[2, 2]

    return H_fitted, K_fitted, float(scale)


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

    all_frames = json.loads(poses_file.read_text(encoding="utf-8"))
    if not all_frames:
        raise RuntimeError(f"No camera frames found in {poses_file}")

    raw_corrections, used_corrections, trusted_tilts = (
        regularized_leveling_corrections(
            all_frames,
            args.max_trusted_tilt,
            args.smooth_window,
        )
    )

    if args.start < 0:
        raise ValueError("--start must be at least 0")
    stop = len(all_frames)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        stop = min(stop, args.start + args.limit)

    frames = all_frames[args.start:stop]
    raw_corrections = raw_corrections[args.start:stop]
    used_corrections = used_corrections[args.start:stop]
    trusted_tilts = trusted_tilts[args.start:stop]

    if not frames:
        raise ValueError("The selected --start/--limit range contains no frames")

    output_images.mkdir(parents=True, exist_ok=True)
    output_homographies.mkdir(parents=True, exist_ok=True)

    rectified_frames = []

    selected = zip(
        frames,
        raw_corrections,
        used_corrections,
        trusted_tilts,
    )

    for number, (frame, raw_correction, used_correction, tilt_trusted) in enumerate(
        selected,
        start=1,
    ):
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

        Q_local = rotvec_to_rotation_matrix(used_correction)
        R_rectified = R_source @ Q_local
        raw_leveling_degrees = float(np.degrees(np.linalg.norm(raw_correction)))
        leveling_degrees = float(np.degrees(np.linalg.norm(used_correction)))
        H = rotation_homography(K, R_source, R_rectified)
        H, K_rectified, fit_scale = fit_homography_to_canvas(
            H,
            K,
            width,
            height,
            args.fit_margin,
        )

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
            "raw_pose_leveling_degrees": raw_leveling_degrees,
            "leveling_rotation_degrees": leveling_degrees,
            "pose_tilt_trusted": bool(tilt_trusted),
            "auto_fit_scale": fit_scale,
            "K": K_rectified.tolist(),
            "fx": float(K_rectified[0, 0]),
            "fy": float(K_rectified[1, 1]),
            "cx": float(K_rectified[0, 2]),
            "cy": float(K_rectified[1, 2]),
            "position_world": position_world.tolist(),
            "rotation_matrix": R_rectified.tolist(),
            "R_camera_to_world": R_rectified.tolist(),
            "R_world_to_camera": R_rectified.T.tolist(),
            "T_camera_to_world": T_camera_to_world.tolist(),
            "T_world_to_camera": T_world_to_camera.tolist(),
        })
        rectified_frames.append(rectified_frame)

        print(
            f"[{number}/{len(frames)}] {dataset_name} "
            f"(raw {raw_leveling_degrees:.2f}, "
            f"used {leveling_degrees:.2f} degrees, "
            f"fit {fit_scale:.3f}x, "
            f"{'trusted' if tilt_trusted else 'regularized'})"
        )

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
            "Images and cameras robustly leveled in a Y-up world. Large "
            "pose-tilt outliers are replaced by temporally regularized "
            "pitch/roll corrections."
        ),
        "pose_convention": (
            "T_camera_to_world maps rectified camera coordinates to "
            "Y-up world coordinates."
        ),
        "image_source": (
            "Each image was warped from its COLMAP-undistorted source image."
        ),
        "rectification": (
            "Local pitch/roll corrections are robustly regularized over time; "
            "large pose tilts are interpolated from trusted neighboring "
            "frames while yaw is left untouched. A per-image 2D similarity "
            "fits the transformed source corners inside the original canvas; "
            "K includes this fit."
        ),
        "max_trusted_pose_tilt_degrees": args.max_trusted_tilt,
        "smoothing_window_frames": args.smooth_window,
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
