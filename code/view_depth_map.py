import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid,
            delimiter="&",
            max_rows=1,
            usecols=(0, 1, 2),
            dtype=int,
        )

        fid.seek(0)

        num_delimiters = 0
        while num_delimiters < 3:
            if fid.read(1) == b"&":
                num_delimiters += 1

        array = np.fromfile(
            fid,
            np.float32,
        )

    array = array.reshape(
        (width, height, channels),
        order="F",
    )

    return np.transpose(array, (1, 0, 2)).squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_map")
    args = parser.parse_args()

    path = Path(args.depth_map)

    depth = read_array(path)

    valid = np.isfinite(depth) & (depth > 0)

    print(f"Shape: {depth.shape}")
    print(f"Valid pixels: {valid.sum()} / {depth.size}")

    if valid.any():
        print(f"Min depth: {depth[valid].min():.3f}")
        print(f"Median depth: {np.median(depth[valid]):.3f}")
        print(f"Max depth: {depth[valid].max():.3f}")

    # Ignore extreme values when visualizing.
    display = depth.copy()
    display[~valid] = np.nan

    if valid.any():
        upper = np.percentile(depth[valid], 95)
        display[display > upper] = upper

    plt.figure(figsize=(12, 7))
    plt.imshow(display)
    plt.colorbar(label="Depth")
    plt.title(path.name)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()