from pathlib import Path

import cv2
import numpy as np

from rectification import rectification_image_from_rot_K
from undistort_and_rectify import (
    read_colmap_camera,
    read_rotation_from_images_txt,
)

# ========= KONFIGURATION =========

IMAGE_DIR = Path("dense/images")
OUTPUT_DIR = Path("dense/images_rectified")

CAMERAS_TXT = "sparse_txt/cameras.txt"
IMAGES_TXT = "sparse_yup_txt/images.txt"
# ================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, model, width, height, K, dist = read_colmap_camera(
        CAMERAS_TXT
    )

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Found {len(image_paths)} images")

    dist = None

    for idx, image_path in enumerate(image_paths, start=1):

        print(
            f"[{idx}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        im = cv2.imread(str(image_path))

        if im is None:
            print(f"Could not read {image_path}")
            continue

        # Undistort
        if dist is not None:
            im = cv2.undistort(im, K, dist)

        # Camera rotation from COLMAP
        R = read_rotation_from_images_txt(
            IMAGES_TXT,
            image_path
        )

        # Rectify
        im_rect, H = rectification_image_from_rot_K(
            im,
            R,
            K
        )

        # Save image
        output_image = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_image), im_rect)

        # Optional: save homography
        output_H = OUTPUT_DIR / (
            image_path.stem + ".H.txt"
        )

        np.savetxt(output_H, H)

    print()
    print(f"Done.")
    print(f"Images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()