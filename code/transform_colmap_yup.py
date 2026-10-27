import argparse
import shutil
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument(
        "--no-auto-flip",
        action="store_true",
        help=(
            "Do not use camera orientations to correct an upside-down "
            "ground-plane orientation."
        ),
    )
    parser.add_argument(
        "--no-up-refinement",
        action="store_true",
        help="Do not fine-tune world up from the robust median camera-up.",
    )
    parser.add_argument(
        "--max-up-refinement-deg",
        type=float,
        default=5.0,
        help=(
            "Maximum automatic camera-based up refinement. "
            "Default: 5 degrees."
        ),
    )
    return parser.parse_args()


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=float)


def rotmat2qvec(R):
    R = np.asarray(R, dtype=float)
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    q /= np.linalg.norm(q)
    return q


def load_matrix(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix in {path}, got {matrix.shape}")
    return matrix


def read_camera_up_vectors(frames_file):
    """Read image-up directions in the original COLMAP world coordinates."""
    up_vectors = []
    for line in frames_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 10:
            continue

        try:
            int(parts[0])
            int(parts[1])
            q_rig_from_world = np.array(list(map(float, parts[2:6])))
            list(map(float, parts[6:9]))
        except ValueError:
            # The second line belonging to a frame contains sensor data,
            # not another pose.
            continue

        R_rig_from_world = qvec2rotmat(q_rig_from_world)
        R_world_from_rig = R_rig_from_world.T

        # COLMAP camera coordinates use +X right, +Y down and +Z forward.
        # Therefore image-up is the negative camera/rig Y axis in world
        # coordinates. The reference sensor is the rig for our single-camera
        # reconstructions.
        up_world = -R_world_from_rig[:, 1]
        up_vectors.append(up_world)

    return np.asarray(up_vectors, dtype=float)


def normalized_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12
    return vectors[valid] / lengths[valid, None]


def robust_median_direction(vectors):
    """Return a robust central unit direction and its inlier count."""
    vectors = normalized_rows(vectors)
    if len(vectors) == 0:
        raise ValueError("No valid direction vectors")

    center = np.median(vectors, axis=0)
    center /= np.linalg.norm(center)

    angles = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
    median_angle = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median_angle)))
    limit = median_angle + max(1.0, 3.0 * 1.4826 * mad)
    inliers = vectors[angles <= limit]

    center = np.median(inliers, axis=0)
    center /= np.linalg.norm(center)
    return center, len(inliers), len(vectors)


def rotation_aligning_vectors(source, target):
    """Return a proper 3x3 rotation mapping source onto target."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))

    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        raise ValueError("Cannot choose a unique rotation for opposite vectors")

    x, y, z = cross
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + K + (K @ K) * ((1.0 - cosine) / (sine * sine))


def orient_ground_up(
    T_world,
    frames_file,
    auto_flip=True,
    refine_up=True,
    max_refinement_degrees=5.0,
):
    """Resolve the up sign and optionally fine-tune it from camera poses."""
    up_vectors = read_camera_up_vectors(frames_file)
    if len(up_vectors) == 0:
        print(
            "WARNING: Could not read camera orientations; "
            "the ground orientation was not checked."
        )
        return T_world

    transformed_up = normalized_rows(
        (T_world[:3, :3] @ up_vectors.T).T
    )
    up_y = transformed_up[:, 1]
    median_up_y = float(np.median(up_y))
    upright = int(np.count_nonzero(up_y > 0))

    print(
        "Ground-orientation check: "
        f"median camera-up dot +Y = {median_up_y:.6f}; "
        f"{upright}/{len(up_y)} cameras have image-up toward +Y."
    )

    if auto_flip and median_up_y < 0:
        # A 180-degree rotation around X reverses Y and Z while preserving
        # a right-handed coordinate system.
        flip_y_and_z = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=float)
        T_world = flip_y_and_z @ T_world
        print(
            "Detected an upside-down ground orientation. "
            "Applied a 180-degree rotation around the X axis."
        )
    elif auto_flip and abs(median_up_y) < 0.05:
        print(
            "WARNING: Camera orientations are nearly perpendicular to Y, so "
            "the ground direction is uncertain. No automatic flip was applied."
        )
    elif auto_flip:
        print("Ground orientation is already correct; no flip was needed.")

    if not refine_up:
        return T_world

    if max_refinement_degrees < 0:
        raise ValueError("--max-up-refinement-deg must be at least 0")

    transformed_up = normalized_rows(
        (T_world[:3, :3] @ up_vectors.T).T
    )
    median_up, inliers, total = robust_median_direction(transformed_up)
    world_up = np.array([0.0, 1.0, 0.0])
    refinement_degrees = float(np.degrees(np.arccos(
        np.clip(median_up @ world_up, -1.0, 1.0)
    )))

    print(
        "Camera-up refinement check: "
        f"robust median differs from +Y by {refinement_degrees:.3f} degrees "
        f"({inliers}/{total} inliers)."
    )

    if refinement_degrees < 1e-6:
        print("No camera-up refinement was needed.")
        return T_world

    if refinement_degrees > max_refinement_degrees:
        print(
            "WARNING: Required up refinement exceeds "
            f"{max_refinement_degrees:.3f} degrees; no refinement was applied."
        )
        return T_world

    refinement = np.eye(4)
    refinement[:3, :3] = rotation_aligning_vectors(median_up, world_up)
    T_world = refinement @ T_world
    print(
        f"Applied camera-based up refinement: {refinement_degrees:.3f} degrees."
    )

    return T_world


def transform_frames(input_file, output_file, R_world, t_world):
    lines = input_file.read_text().splitlines(True)
    with open(output_file, "w") as g:
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                g.write(line)
                continue
            parts = stripped.split()
            if len(parts) < 10:
                g.write(line)
                continue
            try:
                int(parts[0]); int(parts[1])
            except ValueError:
                g.write(line)
                continue
            q_old = np.array(list(map(float, parts[2:6])))
            t_old = np.array(list(map(float, parts[6:9])))
            Rrw_old = qvec2rotmat(q_old)
            Rrw_new = Rrw_old @ R_world.T
            t_new = t_old - Rrw_new @ t_world
            q_new = rotmat2qvec(Rrw_new)
            parts[2:6] = [f"{v:.12f}" for v in q_new]
            parts[6:9] = [f"{v:.12f}" for v in t_new]
            g.write(" ".join(parts) + "\n")


def transform_points(input_file, output_file, R_world, t_world):
    with open(input_file, "r") as f, open(output_file, "w") as g:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                g.write(line)
                continue
            parts = stripped.split()
            if len(parts) < 4:
                g.write(line)
                continue
            try:
                int(parts[0])
            except ValueError:
                g.write(line)
                continue
            xyz = np.array(list(map(float, parts[1:4])))
            xyz_new = R_world @ xyz + t_world
            parts[1:4] = [f"{v:.12f}" for v in xyz_new]
            g.write(" ".join(parts) + "\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "datasets" / args.dataset / "reconstructions" / args.run / "colmap"
    inp = run_root / "sparse_txt"
    out = run_root / "sparse_yup_txt"
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path
    if not inp.exists():
        raise FileNotFoundError(f"Could not find COLMAP TXT model: {inp}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Could not find transformation matrix: {matrix_path}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    T_cc = load_matrix(matrix_path)
    T_z_to_y = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    T_world = T_z_to_y @ T_cc
    frames_in = inp / "frames.txt"
    if frames_in.exists():
        T_world = orient_ground_up(
            T_world,
            frames_in,
            auto_flip=not args.no_auto_flip,
            refine_up=not args.no_up_refinement,
            max_refinement_degrees=args.max_up_refinement_deg,
        )
    R_world = T_world[:3, :3]
    t_world = T_world[:3, 3]

    for name in ["cameras.txt", "rigs.txt", "images.txt"]:
        source = inp / name
        if source.exists():
            (out / name).write_text(source.read_text())

    if frames_in.exists():
        transform_frames(frames_in, out / "frames.txt", R_world, t_world)

    transform_points(inp / "points3D.txt", out / "points3D.txt", R_world, t_world)
    np.savetxt(out / "world_transform_yup.txt", T_world, fmt="%.12f")
    print("Y-up COLMAP model written to:", out)


if __name__ == "__main__":
    main()
