import argparse
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLMAP = shutil.which("colmap")


def run(cmd):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes", "j", "ja"):
            return True
        if answer in ("n", "no", "nej"):
            return False
        print("Svar y eller n.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resumable video -> COLMAP -> Y-up pipeline."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name under datasets/.",
    )

    parser.add_argument(
        "--video",
        help=(
            "Video path, absolute or relative to datasets/<dataset>/. "
            "Not required when using --resume."
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frames per second. Default: 3.",
    )

    parser.add_argument(
        "--colmap",
        default=DEFAULT_COLMAP,
        help="COLMAP executable path. Defaults to the colmap found in PATH.",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help=(
            "Resume an existing run. Use --resume for the latest run, "
            "or --resume RUN_ID for a specific timestamped run."
        ),
    )

    return parser.parse_args()


def frame_number(name):
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def resolve_video(dataset_root, video_arg):
    if not video_arg:
        return None

    video_path = Path(video_arg)
    if not video_path.is_absolute():
        video_path = dataset_root / video_path

    return video_path


def find_run(dataset_root, resume_arg):
    recon_root = dataset_root / "reconstructions"

    if not recon_root.exists():
        raise FileNotFoundError(
            f"No reconstruction directory found: {recon_root}"
        )

    if resume_arg == "latest":
        runs = sorted(
            [p for p in recon_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        if not runs:
            raise FileNotFoundError(
                f"No existing runs found under: {recon_root}"
            )
        return runs[0]

    run_root = recon_root / resume_arg
    if not run_root.exists():
        raise FileNotFoundError(
            f"Requested run does not exist: {run_root}"
        )

    return run_root


def choose_or_create_run(dataset_root, video_path, resume_arg):
    recon_root = dataset_root / "reconstructions"
    recon_root.mkdir(parents=True, exist_ok=True)

    # Resume an existing run.
    if resume_arg:
        run_root = find_run(dataset_root, resume_arg)
        print(f"\nResuming existing run: {run_root}")
        return run_root

    if video_path is None:
        raise ValueError(
            "--video is required when starting a new run."
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Example:
    # video2.mp4 -> video2
    video_name = video_path.stem

    # Find existing versions:
    # video2_v1, video2_v2, ...
    pattern = re.compile(
        rf"^{re.escape(video_name)}_v(\d+)$"
    )

    versions = []

    for path in recon_root.iterdir():
        if not path.is_dir():
            continue

        match = pattern.match(path.name)

        if match:
            versions.append(int(match.group(1)))

    # Pick next available version.
    next_version = max(versions, default=0) + 1

    run_name = f"{video_name}_v{next_version}"

    run_root = recon_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)

    print(
        f"\nCreated new reconstruction: {run_name}"
    )

    return run_root


def list_images(images_dir):
    return sorted(
        [p for p in images_dir.glob("*.jpg") if p.is_file()],
        key=lambda p: frame_number(p.name),
    )


def has_feature_database(database):
    return database.exists() and database.stat().st_size > 0


def database_has_table(database, table_name):
    if not database.exists():
        return False

    try:
        with sqlite3.connect(database) as con:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return row is not None
    except sqlite3.DatabaseError:
        return False


def database_has_features(database):
    if not database_has_table(database, "keypoints"):
        return False

    try:
        with sqlite3.connect(database) as con:
            row = con.execute("SELECT COUNT(*) FROM keypoints").fetchone()
            return bool(row and row[0] > 0)
    except sqlite3.DatabaseError:
        return False


def database_has_matches(database):
    if not database_has_table(database, "matches"):
        return False

    try:
        with sqlite3.connect(database) as con:
            row = con.execute("SELECT COUNT(*) FROM matches").fetchone()
            return bool(row and row[0] > 0)
    except sqlite3.DatabaseError:
        return False


def count_registered_images(model_dir, colmap_bin, temp_root):
    result = subprocess.run(
        [
            str(colmap_bin),
            "model_analyzer",
            "--path",
            str(model_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"Registered images:\s*(\d+)",
        result.stdout + result.stderr,
    )

    if match is None:
        raise RuntimeError(
            f"Could not determine registered image count for {model_dir}"
        )

    return int(match.group(1))


def choose_best_model(sparse_dir, colmap_bin):
    models = [
        p
        for p in sparse_dir.iterdir()
        if p.is_dir() and (p / "images.bin").exists()
    ]

    if not models:
        raise RuntimeError("No sparse COLMAP models found.")

    temp_root = sparse_dir / "_count_tmp"
    temp_root.mkdir(exist_ok=True)
    scored = []

    try:
        for model in models:
            count = count_registered_images(
                model,
                colmap_bin,
                temp_root,
            )
            scored.append((count, model))
            print(
                f"Model {model.name}: "
                f"{count} registered images"
            )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][0]


def sparse_models_exist(sparse_dir):
    if not sparse_dir.exists():
        return False

    return any(
        p.is_dir() and (p / "images.bin").exists()
        for p in sparse_dir.iterdir()
    )


def sparse_txt_ready(sparse_txt):
    required = [
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ]
    return all((sparse_txt / name).exists() for name in required)


def dense_workspace_ready(dense_dir):
    required = [
        dense_dir / "images",
        dense_dir / "sparse",
    ]
    return all(path.exists() for path in required)


def dense_fused_ready(dense_dir):
    return (dense_dir / "fused.ply").exists()


def aligned_sparse_ready(sparse_yup):
    required = [
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "frames.txt",
        "world_transform_yup.txt",
    ]
    return all((sparse_yup / name).exists() for name in required)


def aligned_dense_ready(dense_dir):
    return (dense_dir / "fused_yup.ply").exists()


def exported_poses_ready(reconstruction_dir):
    required = [
        "metadata.json",
        "camera_poses.json",
        "relative_poses.json",
    ]
    return all((reconstruction_dir / name).exists() for name in required)


def trajectory_ready(reconstruction_dir):
    return (reconstruction_dir / "camera_trajectory.ply").exists()


def sparse_quality_check(registered, total):
    if total == 0:
        raise RuntimeError("No input frames found.")

    coverage = registered / total
    percent = coverage * 100

    if coverage >= 0.80:
        quality = "GOOD"
    elif coverage >= 0.50:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    print("\n" + "=" * 60)
    print("SPARSE RECONSTRUCTION QUALITY")
    print("=" * 60)
    print(f"Input frames:       {total}")
    print(f"Registered frames:  {registered}")
    print(f"Coverage:           {percent:.1f}%")
    print(f"Quality:            {quality}")
    print("=" * 60)

    return coverage, quality


def main():
    args = parse_args()
    if args.colmap is None:
        raise FileNotFoundError(
            "Could not find COLMAP in PATH. Run: which colmap"
        )

    colmap_bin = Path(args.colmap).expanduser().resolve()

    if not colmap_bin.is_file():
        raise FileNotFoundError(
            f"COLMAP executable not found: {colmap_bin}"
        )

    dataset_root = REPO_ROOT / "datasets" / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    video_path = resolve_video(dataset_root, args.video)

    if args.resume and video_path and not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    run_root = choose_or_create_run(
        dataset_root,
        video_path,
        args.resume,
    )

    images_dir = run_root / "images"
    colmap_dir = run_root / "colmap"
    database = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_txt = colmap_dir / "sparse_txt"
    dense_dir = colmap_dir / "dense"
    sparse_yup = colmap_dir / "sparse_yup_txt"
    reconstruction_dir = run_root / "reconstruction"
    matrix_copy = run_root / "transformation.txt"

    images_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PARSE FOREST PIPELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if video_path:
        print(f"Video:   {video_path}")
    print(f"Run:     {run_root}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Frames
    # ---------------------------------------------------------
    images = list_images(images_dir)

    if images:
        print(
            f"[SKIP] Frames already exist: {len(images)} images"
        )
    else:
        if video_path is None:
            raise ValueError(
                "No frames found and no --video was provided. "
                "Provide --video to resume from the frame step."
            )

        run([
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={args.fps}",
            images_dir / "%06d.jpg",
        ])

        images = list_images(images_dir)
        print(f"Extracted {len(images)} frames.")

    if not images:
        raise RuntimeError("No JPEG frames found.")

    # ---------------------------------------------------------
    # 2. Feature extraction
    # ---------------------------------------------------------
    if database_has_features(database):
        print("[SKIP] Feature extraction already present in database.")
    else:
        run([
            colmap_bin,
            "feature_extractor",
            "--database_path",
            database,
            "--image_path",
            images_dir,
        ])

    # ---------------------------------------------------------
    # 3. Sequential matching
    # ---------------------------------------------------------
    if database_has_matches(database):
        print("[SKIP] Matches already present in database.")
    else:
        run([
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            database,
            "--SequentialMatching.overlap",
            "20",
            "--SequentialMatching.loop_detection",
            "1",
            "--SequentialMatching.loop_detection_num_nearest_neighbors",
            "5",
        ])

    # ---------------------------------------------------------
    # 4. Sparse reconstruction
    # ---------------------------------------------------------
    if sparse_models_exist(sparse_dir):
        print("[SKIP] Sparse models already exist.")
    else:
        sparse_dir.mkdir(parents=True, exist_ok=True)
        run([
            colmap_bin,
            "mapper",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--output_path",
            sparse_dir,
        ])

    best_model, registered = choose_best_model(
        sparse_dir,
        colmap_bin,
    )

    print(
        f"\nSelected sparse model {best_model.name} "
        f"with {registered} registered images."
    )

    coverage, quality = sparse_quality_check(
    registered,
    len(images),
)

    if quality == "POOR":
        print(
            "\nWARNING: Less than 50% of the input frames were "
            "registered in the best sparse model."
        )
        print(
            "Dense reconstruction is not recommended "
            "before inspecting the sparse model."
        )

        if not ask_yes_no(
            "Continue anyway?"
        ):
            print(
                f"\nStopped after sparse reconstruction.\n"
                f"Run directory: {run_root}"
            )
            return

    # ---------------------------------------------------------
    # 5. Sparse TXT conversion
    # ---------------------------------------------------------
    if sparse_txt_ready(sparse_txt):
        print("[SKIP] sparse_txt already exists.")
    else:
        sparse_txt.mkdir(parents=True, exist_ok=True)
        run([
            colmap_bin,
            "model_converter",
            "--input_path",
            best_model,
            "--output_path",
            sparse_txt,
            "--output_type",
            "TXT",
        ])

    print(
        f"\nSparse reconstruction is ready."
    )

    if quality == "GOOD":
        print(
            "Sparse coverage is good. "
            "Dense reconstruction is recommended."
        )
    else:
        print(
            "Sparse coverage is acceptable. "
            "Inspect the sparse model before dense reconstruction."
        )

    if not ask_yes_no("Run or resume dense reconstruction?"):
        return

    # ---------------------------------------------------------
    # 6. Dense reconstruction
    # ---------------------------------------------------------
    dense_dir.mkdir(parents=True, exist_ok=True)

    if dense_fused_ready(dense_dir):
        print("[SKIP] Dense fused point cloud already exists.")
    else:
        if dense_workspace_ready(dense_dir):
            print("[SKIP] Dense workspace already undistorted.")
        else:
            run([
                colmap_bin,
                "image_undistorter",
                "--image_path",
                images_dir,
                "--input_path",
                best_model,
                "--output_path",
                dense_dir,
                "--output_type",
                "COLMAP",
            ])

        depth_maps = dense_dir / "stereo" / "depth_maps"
        if depth_maps.exists() and any(depth_maps.iterdir()):
            print("[SKIP] PatchMatch depth maps already exist.")
        else:
            run([
                colmap_bin,
                "patch_match_stereo",
                "--workspace_path",
                dense_dir,
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ])

        fused = dense_dir / "fused.ply"
        if fused.exists():
            print("[SKIP] Stereo fusion already exists.")
        else:
            run([
                colmap_bin,
                "stereo_fusion",
                "--workspace_path",
                dense_dir,
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_path",
                fused,
            ])

    fused = dense_dir / "fused.ply"
    print(f"\nDense reconstruction is ready:\n{fused}")

    # ---------------------------------------------------------
    # 7. CloudCompare alignment
    # ---------------------------------------------------------
    if matrix_copy.exists():
        print(f"[SKIP] Transformation matrix already stored: {matrix_copy}")
    else:
        matrix_input = input(
            "Path to CloudCompare transformation matrix "
            "[Enter to stop before alignment]: "
        ).strip()

        if not matrix_input:
            print(
                f"\nStopped after dense reconstruction.\n"
                f"Run directory: {run_root}"
            )
            return

        matrix_path = Path(matrix_input)
        if not matrix_path.is_absolute():
            matrix_path = REPO_ROOT / matrix_path

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Transformation matrix not found: {matrix_path}"
            )

        shutil.copy(matrix_path, matrix_copy)
        print(f"Saved matrix to: {matrix_copy}")

    # ---------------------------------------------------------
    # 8. Y-up sparse model
    # ---------------------------------------------------------
    if aligned_sparse_ready(sparse_yup):
        print("[SKIP] Y-up sparse model already exists.")
    else:
        run([
            "python",
            REPO_ROOT / "code" / "transform_colmap_yup.py",
            "--dataset",
            args.dataset,
            "--run",
            run_root.name,
            "--matrix",
            matrix_copy,
        ])

    # ---------------------------------------------------------
    # 9. Y-up dense point cloud
    # ---------------------------------------------------------
    fused_yup = dense_dir / "fused_yup.ply"
    if fused_yup.exists():
        print("[SKIP] Y-up dense point cloud already exists.")
    else:
        run([
            "python",
            REPO_ROOT / "code" / "transform_pointcloud.py",
            "--dataset",
            args.dataset,
            "--run",
            run_root.name,
            "--matrix",
            sparse_yup / "world_transform_yup.txt",
        ])

    # ---------------------------------------------------------
    # 10. Export camera poses
    # ---------------------------------------------------------
    if exported_poses_ready(reconstruction_dir):
        print("[SKIP] Camera pose export already exists.")
    else:
        run([
            "python",
            REPO_ROOT / "code" / "export_dataset.py",
            "--dataset",
            args.dataset,
            "--run",
            run_root.name,
        ])

    # ---------------------------------------------------------
    # 11. Camera trajectory
    # ---------------------------------------------------------
    if trajectory_ready(reconstruction_dir):
        print("[SKIP] Camera trajectory already exists.")
    else:
        run([
            "python",
            REPO_ROOT / "code" / "export_camera_trajectory.py",
            "--dataset",
            args.dataset,
            "--run",
            run_root.name,
        ])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Run:               {run_root}")
    print(f"Sparse:            {best_model}")
    print(f"Y-up sparse:       {sparse_yup}")
    print(f"Y-up dense:        {fused_yup}")
    print(f"Camera poses:      {reconstruction_dir / 'camera_poses.json'}")
    print(f"Camera trajectory: {reconstruction_dir / 'camera_trajectory.ply'}")
    print("=" * 60)


if __name__ == "__main__":
    main()