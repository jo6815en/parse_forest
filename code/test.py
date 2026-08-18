from pathlib import Path
import numpy as np


def qvec2rotmat(q):
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,
         2*qx*qy - 2*qz*qw,
         2*qx*qz + 2*qy*qw],

        [2*qx*qy + 2*qz*qw,
         1 - 2*qx*qx - 2*qz*qz,
         2*qy*qz - 2*qx*qw],

        [2*qx*qz - 2*qy*qw,
         2*qy*qz + 2*qx*qw,
         1 - 2*qx*qx - 2*qy*qy]
    ])


IMAGE_NAME = "frame_00001.jpg"
IMAGES_TXT = "sparse_yup_txt/images.txt"


with open(IMAGES_TXT, "r") as f:
    lines = f.readlines()

q = None

for line in lines:
    line = line.strip()

    if not line or line.startswith("#"):
        continue

    parts = line.split()

    if len(parts) >= 10:
        name = Path(parts[9]).name

        if name == IMAGE_NAME:
            q = np.array(list(map(float, parts[1:5])))
            break

if q is None:
    raise RuntimeError(f"Could not find {IMAGE_NAME}")

print("Quaternion:")
print(q)

R_cw = qvec2rotmat(q)

print("\nR_cw (world -> camera):")
print(R_cw)

R_wc = R_cw.T

print("\nR_wc (camera -> world):")
print(R_wc)

x_cam = R_wc[:, 0]
y_cam = R_wc[:, 1]
z_cam = R_wc[:, 2]

print("\nCamera axes expressed in world coordinates:")
print("x_cam =", x_cam)
print("y_cam =", y_cam)
print("z_cam =", z_cam)

world_up = np.array([0, 1, 0])

print("\nDot products with world up [0,1,0]:")
print("x·up =", np.dot(x_cam, world_up))
print("y·up =", np.dot(y_cam, world_up))
print("z·up =", np.dot(z_cam, world_up))