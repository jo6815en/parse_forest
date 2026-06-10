import numpy as np
import cv2


def null_space_2d(v):
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    _, _, vh = np.linalg.svd(v.reshape(1, 3))
    return vh[1:].T


def rectification_image_from_rot_K(im, R, K):
    M, N = im.shape[:2]

    r2 = R[:, 1]
    r2 = r2 / np.linalg.norm(r2)

    v2 = null_space_2d(r2)

    rr3 = v2 @ (np.array([0, 0, 1.0]) @ v2).T
    rr3 = rr3 / np.linalg.norm(rr3)

    rr1 = np.cross(r2, rr3)
    rr1 = rr1 / np.linalg.norm(rr1)

    R0 = np.vstack([rr1, r2, rr3])

    H = K @ R0 @ np.linalg.inv(K)

    xc = np.array([N / 2, M / 2, 1.0])
    xcp = H @ xc
    xcp = xcp / xcp[2]

    T = np.array([
        [1, 0, N / 2 - xcp[0]],
        [0, 1, M / 2 - xcp[1]],
        [0, 0, 1]
    ], dtype=float)

    H_out = T @ H

    imout = cv2.warpPerspective(
        im,
        H_out,
        dsize=(N, M),
        flags=cv2.INTER_LINEAR
    )

    return imout, H_out