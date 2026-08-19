import argparse
import shutil
import subprocess
from pathlib import Path


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


def run(cmd, cwd=None):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full COLMAP reconstruction pipeline."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder under datasets/, e.g. campus",
    )
    return parser.parse_args()


def count_registered_images(model_dir):
    """
    Convert the model to TXT temporarily and count registered images.
    """
    temp_dir = model_dir.parent / f"{model_dir.name}_txt"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    run([
        str(COLMAP_BIN),
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


def choose_best_model(sparse_dir):
    models = [
        p for p in sparse_dir.iterdir()
        if p.is_dir() and (p / "images.bin").exists()
    ]

    if not models:
        raise RuntimeError("No COLMAP sparse models found.")

    scored = []

    for model in models:
        count = count_registered_images(model)
        scored.append((count, model))
        print(f"Model {model.name}: {count} registered images")

    scored.sort(reverse=True, key=lambda x: x[0])

    best_count, best_model = scored[0]

    print(
        f"\nSelected model {best_model.name} "
        f"with {best_count} registered images."
    )

    return best_model


def main():
    args = parse_args()

    dataset_root = REPO_ROOT / "datasets" / args.dataset
    image_dir = dataset_root / "images"
    colmap_dir = dataset_root / "colmap"

    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_txt_dir = colmap_dir / "sparse_txt"
    dense_dir = colmap_dir / "dense"

    if not COLMAP_BIN.exists():
        raise FileNotFoundError(
            f"COLMAP executable not found:\n{COLMAP_BIN}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    colmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Images:  {image_dir}")
    print(f"COLMAP:  {COLMAP_BIN}")

    # ---------------------------------------------------------
    # 1. Feature extraction
    # ---------------------------------------------------------
    if database_path.exists():
        print("\nRemoving existing COLMAP database.")
        database_path.unlink()

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
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)

    sparse_dir.mkdir(parents=True)

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
    # 4. Select largest reconstruction
    # ---------------------------------------------------------
    best_model = choose_best_model(sparse_dir)

    # ---------------------------------------------------------
    # 5. Export selected model to TXT
    # ---------------------------------------------------------
    if sparse_txt_dir.exists():
        shutil.rmtree(sparse_txt_dir)

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
    # 6. Dense reconstruction: image undistortion
    # ---------------------------------------------------------
    if dense_dir.exists():
        shutil.rmtree(dense_dir)

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
    # 7. Dense reconstruction: PatchMatch Stereo
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
    # 8. Dense reconstruction: stereo fusion
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

    # ---------------------------------------------------------
    # 9. Export poses / dataset
    # ---------------------------------------------------------
    export_script = REPO_ROOT / "code" / "export_dataset.py"

    run([
        "python",
        str(export_script),
        "--dataset",
        args.dataset,
    ])

    print("\n" + "=" * 60)
    print("COLMAP reconstruction complete")
    print("=" * 60)
    print(f"Sparse model: {best_model}")
    print(f"Dense cloud:  {fused_path}")
    print(f"Output:       {dataset_root / 'reconstruction'}")


if __name__ == "__main__":
    main()
