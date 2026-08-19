import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

COLMAP_BIN = (
    Path.home()
    / "Doktorand"
    / "colmap-cuda"
    / "build"
    / "src"
    / "colmap"
    / "exe"
    / "colmap"
)


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()

        if answer in ("y", "yes", "j", "ja"):
            return True

        if answer in ("n", "no", "nej"):
            return False

        print("Svara y eller n.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a COLMAP reconstruction as a timestamped experiment."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder under datasets/, e.g. campus",
    )

    return parser.parse_args()


def qvec2rotmat(q):
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
    R = qvec2rotmat(q)
    t = np.asarray(t, dtype=float)

    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t

    return T


def rotation_matrix_to_quaternion(R):
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


def parse_images_txt(images_txt):
    poses = {}

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
            int(parts[0])
        except ValueError:
            i += 1
            continue

        q = list(map(float, parts[1:5]))
        t = list(map(float, parts[5:8]))
        camera_id = int(parts[8])
        name = parts[9]

        T = camera_to_world(q, t)

        poses[name] = {
            "camera_id": camera_id,
            "T_camera_to_world": T,
        }

        i += 2

    return poses


def frame_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def export_camera_data(images_txt, images_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    poses = parse_images_txt(images_txt)
    names = sorted(poses.keys(), key=frame_number)

    metadata = {
        "pose_convention": (
            "T_camera_to_world maps camera coordinates "
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

        source = images_dir / name
        target = output_images_dir / new_name

        if not source.exists():
            raise FileNotFoundError(f"Missing image: {source}")

        shutil.copy(source, target)

        T = poses[name]["T_camera_to_world"]
        R = T[:3, :3]
        position = T[:3, 3]
        quaternion = rotation_matrix_to_quaternion(R)

        frame = {
            "index": idx,
            "original_name": name,
            "dataset_name": new_name,
            "camera_id": poses[name]["camera_id"],
            "position_world": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "rotation_matrix": R.tolist(),
            "T_camera_to_world": T.tolist(),
        }

        metadata["frames"].append(frame)
        camera_poses.append(frame)

    relative = []

    for idx in range(len(names) - 1):
        T_wa = poses[names[idx]]["T_camera_to_world"]
        T_wb = poses[names[idx + 1]]["T_camera_to_world"]

        T_a_to_b = np.linalg.inv(T_wb) @ T_wa

        relative.append({
            "from_index": idx,
            "to_index": idx + 1,
            "from_image": f"{idx:06d}.jpg",
            "to_image": f"{idx + 1:06d}.jpg",
            "original_from": names[idx],
            "original_to": names[idx + 1],
            "T_a_to_b": T_a_to_b.tolist(),
        })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "camera_poses.json", "w") as f:
        json.dump(camera_poses, f, indent=2)

    with open(output_dir / "relative_poses.json", "w") as f:
        json.dump(relative, f, indent=2)

    print(f"Exported {len(names)} camera poses.")


def count_registered_images(model_dir, colmap_bin):
    temp_dir = model_dir.parent / f"{model_dir.name}_txt"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    run([
        str(colmap_bin),
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(temp_dir),
        "--output_type",
        "TXT",
    ])

    images_txt = temp_dir / "images.txt"

    count = 0

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            try:
                int(parts[0])
            except ValueError:
                continue

            count += 1

    shutil.rmtree(temp_dir)

    return count


def choose_best_model(sparse_dir, colmap_bin):
    models = [
        p for p in sparse_dir.iterdir()
        if p.is_dir() and (p / "images.bin").exists()
    ]

    if not models:
        raise RuntimeError("No sparse COLMAP models were found.")

    scored = []

    for model in models:
        count = count_registered_images(model, colmap_bin)
        scored.append((count, model))
        print(f"Model {model.name}: {count} registered images")

    scored.sort(reverse=True, key=lambda x: x[0])

    return scored[0][1], scored[0][0]


def main():
    args = parse_args()

    dataset_root = REPO_ROOT / "datasets" / args.dataset
    image_dir = dataset_root / "images"

    if not COLMAP_BIN.exists():
        raise FileNotFoundError(
            f"COLMAP executable not found:\n{COLMAP_BIN}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    # ---------------------------------------------------------
    # Create a unique run directory
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        dataset_root
        / "reconstructions"
        / timestamp
    )

    colmap_dir = run_dir / "colmap"
    sparse_dir = colmap_dir / "sparse"
    sparse_txt_dir = colmap_dir / "sparse_txt"
    dense_dir = colmap_dir / "dense"
    export_dir = run_dir / "reconstruction"

    run_dir.mkdir(parents=True, exist_ok=False)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    database_path = colmap_dir / "database.db"

    print("\n" + "=" * 60)
    print("COLMAP reconstruction")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Run:     {run_dir}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Feature extraction
    # ---------------------------------------------------------
    run([
        str(COLMAP_BIN),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
    ])

    # ---------------------------------------------------------
    # 2. Sequential matching
    # ---------------------------------------------------------
    run([
        str(COLMAP_BIN),
        "sequential_matcher",
        "--database_path",
        str(database_path),
    ])

    # ---------------------------------------------------------
    # 3. Sparse reconstruction
    # ---------------------------------------------------------
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run([
        str(COLMAP_BIN),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ])

    # ---------------------------------------------------------
    # 4. Select best sparse model
    # ---------------------------------------------------------
    best_model, registered = choose_best_model(
        sparse_dir,
        COLMAP_BIN,
    )

    print("\n" + "=" * 60)
    print("Sparse reconstruction complete")
    print("=" * 60)
    print(f"Best model:       {best_model.name}")
    print(f"Registered images: {registered}")
    print(f"Run directory:     {run_dir}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 5. Export sparse model to TXT
    # ---------------------------------------------------------
    run([
        str(COLMAP_BIN),
        "model_converter",
        "--input_path",
        str(best_model),
        "--output_path",
        str(sparse_txt_dir),
        "--output_type",
        "TXT",
    ])

    # ---------------------------------------------------------
    # Ask whether to continue with dense reconstruction
    # ---------------------------------------------------------
    if not ask_yes_no(
        "\nSparse reconstruction is ready. Run dense reconstruction?"
    ):
        print("\nFinished after sparse reconstruction.")
        print(f"Results saved in:\n{run_dir}")
        return

    # ---------------------------------------------------------
    # 6. Image undistortion
    # ---------------------------------------------------------
    run([
        str(COLMAP_BIN),
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(best_model),
        "--output_path",
        str(dense_dir),
        "--output_type",
        "COLMAP",
    ])

    # ---------------------------------------------------------
    # 7. PatchMatch Stereo
    # ---------------------------------------------------------
    run([
        str(COLMAP_BIN),
        "patch_match_stereo",
        "--workspace_path",
        str(dense_dir),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
    ])

    # ---------------------------------------------------------
    # 8. Stereo fusion
    # ---------------------------------------------------------
    fused_path = dense_dir / "fused.ply"

    run([
        str(COLMAP_BIN),
        "stereo_fusion",
        "--workspace_path",
        str(dense_dir),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_path",
        str(fused_path),
    ])

    print("\nDense reconstruction complete.")
    print(f"Dense point cloud:\n{fused_path}")

    # ---------------------------------------------------------
    # Ask whether to save/export dataset
    # ---------------------------------------------------------
    if not ask_yes_no(
        "\nDense reconstruction is ready. Export camera poses and dataset?"
    ):
        print("\nDense reconstruction kept.")
        print(f"Results saved in:\n{run_dir}")
        return

    # ---------------------------------------------------------
    # 9. Export poses and images
    # ---------------------------------------------------------
    export_camera_data(
        sparse_txt_dir / "images.txt",
        image_dir,
        export_dir,
    )

    # ---------------------------------------------------------
    # 10. Export camera trajectory
    # ---------------------------------------------------------
    trajectory_path = export_dir / "camera_trajectory.ply"

    poses_file = export_dir / "camera_poses.json"

    with open(poses_file, "r") as f:
        camera_poses = json.load(f)

    points = [
        pose["position_world"]
        for pose in camera_poses
    ]

    with open(trajectory_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {max(0, len(points) - 1)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        for point in points:
            f.write(
                f"{point[0]} {point[1]} {point[2]}\n"
            )

        for i in range(len(points) - 1):
            f.write(f"{i} {i + 1}\n")

    print("\n" + "=" * 60)
    print("COLMAP reconstruction complete")
    print("=" * 60)
    print(f"Run directory:       {run_dir}")
    print(f"Sparse model:        {best_model}")
    print(f"Dense point cloud:   {fused_path}")
    print(f"Camera poses:        {export_dir / 'camera_poses.json'}")
    print(f"Relative poses:      {export_dir / 'relative_poses.json'}")
    print(f"Camera trajectory:   {trajectory_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
