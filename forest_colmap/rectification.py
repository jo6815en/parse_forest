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

    center = np.array([N / 2, M / 2, 1.0])

    # Flytta så att originalbildens centrum hamnar i outputbildens centrum
    center_proj = H @ center
    center_proj = center_proj / center_proj[2]

    T_center = np.array([
        [1, 0, N / 2 - center_proj[0]],
        [0, 1, M / 2 - center_proj[1]],
        [0, 0, 1]
    ], dtype=float)

    H_centered = T_center @ H

    # Beräkna lokal skala runt bildcentrum
    p0 = H_centered @ center
    p0 = p0 / p0[2]

    px = H_centered @ np.array([N / 2 + 1, M / 2, 1.0])
    px = px / px[2]

    py = H_centered @ np.array([N / 2, M / 2 + 1, 1.0])
    py = py / py[2]

    sx = np.linalg.norm(px[:2] - p0[:2])
    sy = np.linalg.norm(py[:2] - p0[:2])
    s = (sx + sy) / 2.0

    # Kompensera skalan runt outputbildens centrum
    S = np.array([
        [1 / s, 0, N / 2 * (1 - 1 / s)],
        [0, 1 / s, M / 2 * (1 - 1 / s)],
        [0, 0, 1]
    ], dtype=float)

    H_out = S @ H_centered

    imout = cv2.warpPerspective(
        im,
        H_out,
        dsize=(N, M),
        flags=cv2.INTER_LINEAR
    )

    return imout, H_out