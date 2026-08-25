import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transform a COLMAP model using a CloudCompare matrix "
            "and convert Z-up to Y-up."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder under datasets/, for example: torpa",
    )

    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to a text file containing a 4x4 CloudCompare transformation matrix.",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Reconstruction run, for example: 20260824_142747",
    )

    return parser.parse_args()


def qvec2rotmat(q):
    w, x, y, z = q

    return np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w,
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w,
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y,
        ],
    ])


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
        s = 2.0 * np.sqrt(
            1.0 + R[0, 0] - R[1, 1] - R[2, 2]
        )
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(
            1.0 + R[1, 1] - R[0, 0] - R[2, 2]
        )
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s

    else:
        s = 2.0 * np.sqrt(
            1.0 + R[2, 2] - R[0, 0] - R[1, 1]
        )
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s

    q /= np.linalg.norm(q)

    return q


def load_matrix(path):
    matrix = np.loadtxt(path)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 matrix in {path}, got {matrix.shape}"
        )

    return matrix


def transform_frames(input_file, output_file, R_world, t_world):
    """
    Transform COLMAP frames.txt.

    COLMAP frame pose:
        x_rig = R_rw @ x_world + t_rw

    New world coordinates:
        x_new = R_world @ x_old + t_world

    Therefore:
        R_rw_new = R_rw_old @ R_world.T
        t_rw_new = t_rw_old - R_rw_new @ t_world
    """

    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as g:

        for line in lines:

            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                g.write(line)
                continue

            parts = stripped.split()

            # FRAME_ID RIG_ID QW QX QY QZ TX TY TZ NUM_DATA_IDS ...
            if len(parts) < 10:
                g.write(line)
                continue

            try:
                int(parts[0])
                int(parts[1])
            except ValueError:
                g.write(line)
                continue

            q_old = np.array(
                list(map(float, parts[2:6]))
            )

            t_old = np.array(
                list(map(float, parts[6:9]))
            )

            R_rw_old = qvec2rotmat(q_old)

            R_rw_new = R_rw_old @ R_world.T
            t_new = t_old - R_rw_new @ t_world

            q_new = rotmat2qvec(R_rw_new)

            parts[2:6] = [
                f"{value:.12f}"
                for value in q_new
            ]

            parts[6:9] = [
                f"{value:.12f}"
                for value in t_new
            ]

            g.write(" ".join(parts) + "\n")


def transform_points(input_file, output_file, R_world, t_world):
    """
    Transform COLMAP points3D.txt.
    """

    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as g:

        for line in lines:

            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                g.write(line)
                continue

            parts = stripped.split()

            # POINT3D_ID X Y Z R G B ERROR ...
            if len(parts) < 4:
                g.write(line)
                continue

            try:
                int(parts[0])
            except ValueError:
                g.write(line)
                continue

            xyz_old = np.array(
                list(map(float, parts[1:4]))
            )

            xyz_new = R_world @ xyz_old + t_world

            parts[1:4] = [
                f"{value:.12f}"
                for value in xyz_new
            ]

            g.write(" ".join(parts) + "\n")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    dataset_root = repo_root / "datasets" / args.dataset

    run_root = (
        dataset_root
        / "reconstructions"
        / args.run
        / "colmap"
    )

    inp = run_root / "sparse_txt"
    out = run_root / "sparse_yup_txt_v3"

    matrix_path = Path(args.matrix)

    if not matrix_path.is_absolute():
        matrix_path = repo_root / matrix_path

    if not inp.exists():
        raise FileNotFoundError(
            f"Could not find COLMAP model:\n{inp}"
        )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Could not find transformation matrix:\n{matrix_path}"
        )

    if out.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{out}"
        )

    out.mkdir(parents=True)

    # --------------------------------------------------
    # Load CloudCompare matrix
    # --------------------------------------------------

    T_cc = load_matrix(matrix_path)

    # --------------------------------------------------
    # CloudCompare result is Z-up.
    #
    # Convert:
    #
    # old +Z -> new +Y
    # old +Y -> new -Z
    # old +X -> new +X
    # --------------------------------------------------

    T_z_to_y = np.array([
        [1,  0,  0, 0],
        [0,  0,  1, 0],
        [0, -1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=float)

    # First CloudCompare alignment,
    # then Z-up -> Y-up.
    T_world = T_z_to_y @ T_cc

    R_world = T_world[:3, :3]
    t_world = T_world[:3, 3]

    print("\nCloudCompare matrix:")
    print(T_cc)

    print("\nFinal world transformation (Y-up):")
    print(T_world)

    # --------------------------------------------------
    # Copy files that describe the sensor/rig itself.
    #
    # images.txt contains sensor-from-rig information,
    # so it should NOT be transformed here.
    # --------------------------------------------------

    for name in [
        "cameras.txt",
        "rigs.txt",
        "images.txt",
    ]:
        source = inp / name

        if source.exists():
            (out / name).write_text(
                source.read_text()
            )

    # --------------------------------------------------
    # Transform frames.txt
    # --------------------------------------------------

    frames_in = inp / "frames.txt"
    frames_out = out / "frames.txt"

    if frames_in.exists():
        transform_frames(
            frames_in,
            frames_out,
            R_world,
            t_world,
        )
    else:
        print("Warning: frames.txt not found.")

    # --------------------------------------------------
    # Transform points3D.txt
    # --------------------------------------------------

    transform_points(
        inp / "points3D.txt",
        out / "points3D.txt",
        R_world,
        t_world,
    )

    # --------------------------------------------------
    # Save final transformation
    # --------------------------------------------------

    np.savetxt(
        out / "world_transform_yup.txt",
        T_world,
        fmt="%.12f",
    )

    print("\nDone.")
    print(f"Input model:  {inp}")
    print(f"Output model: {out}")
    print(f"Matrix:       {matrix_path}")
    print(
        f"Saved final transform: "
        f"{out / 'world_transform_yup.txt'}"
    )


if __name__ == "__main__":
    main()