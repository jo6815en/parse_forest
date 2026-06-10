import argparse
from pathlib import Path

import cv2
import numpy as np

from rectification import rectification_image_from_rot_K


def qvec2rotmat(q):
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def read_colmap_camera(camera_txt_path):
    with open(camera_txt_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            f, cx, cy, k = params

            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=float)

            dist = np.array([k, 0, 0, 0], dtype=float)

            return camera_id, model, width, height, K, dist

    raise ValueError(f"No camera found in {camera_txt_path}")


def read_rotation_from_images_txt(images_txt_path, image_path):
    image_name = Path(image_path).name

    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        # COLMAP image line:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
        if len(parts) >= 10:
            colmap_image_name = Path(parts[9]).name

            if colmap_image_name == image_name:
                q = np.array(list(map(float, parts[1:5])), dtype=float)
                return qvec2rotmat(q)

            i += 2
        else:
            i += 1

    raise ValueError(f"Could not find image {image_name} in {images_txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--cameras_txt", required=True)
    parser.add_argument("--images_txt", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if im is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    _, model, width, height, K, dist = read_colmap_camera(args.cameras_txt)

    if im.shape[1] != width or im.shape[0] != height:
        print(
            f"Warning: image size is {im.shape[1]}x{im.shape[0]}, "
            f"but COLMAP camera says {width}x{height}"
        )

    im_undist = cv2.undistort(im, K, dist)

    R = read_rotation_from_images_txt(args.images_txt, image_path)

    im_rect, H = rectification_image_from_rot_K(im_undist, R, K)

    cv2.imwrite(str(output_path), im_rect)

    H_path = output_path.with_suffix(".H.txt")
    np.savetxt(H_path, H)

    print(f"Camera model: {model}")
    print(f"Saved rectified image to: {output_path}")
    print(f"Saved homography to: {H_path}")


if __name__ == "__main__":
    main()