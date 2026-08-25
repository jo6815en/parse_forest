import argparse
import shutil
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--matrix", required=True)
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
    R_world = T_world[:3, :3]
    t_world = T_world[:3, 3]

    for name in ["cameras.txt", "rigs.txt", "images.txt"]:
        source = inp / name
        if source.exists():
            (out / name).write_text(source.read_text())

    frames_in = inp / "frames.txt"
    if frames_in.exists():
        transform_frames(frames_in, out / "frames.txt", R_world, t_world)

    transform_points(inp / "points3D.txt", out / "points3D.txt", R_world, t_world)
    np.savetxt(out / "world_transform_yup.txt", T_world, fmt="%.12f")
    print("Y-up COLMAP model written to:", out)


if __name__ == "__main__":
    main()
