import numpy as np
import re

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qz*qx + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qz*qx - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])

def world_to_camera_T(q, t):
    T = np.eye(4)
    T[:3, :3] = qvec2rotmat(q)
    T[:3, 3] = np.array(t)
    return T

def camera_to_world_T(q, t):
    T_wc = world_to_camera_T(q, t)
    return np.linalg.inv(T_wc)

def frame_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1

images = {}

with open("sparse_txt/images.txt") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    if line.startswith("#") or len(line) == 0:
        i += 1
        continue

    parts = line.split()

    try:
        image_id = int(parts[0])
    except ValueError:
        i += 1
        continue

    q = list(map(float, parts[1:5]))
    t = list(map(float, parts[5:8]))
    name = parts[9]

    images[name] = {
        "image_id": image_id,
        "q": q,
        "t": t,
        "T_cw": camera_to_world_T(q, t)
    }

    i += 2

names = sorted(images.keys(), key=frame_number)

print(f"Loaded {len(names)} poses")
print("Registered frames:")
print(names)

print("\nRelative poses between consecutive registered frames:\n")

for a, b in zip(names[:-1], names[1:]):
    T_a = images[a]["T_cw"]
    T_b = images[b]["T_cw"]

    # pose from camera a to camera b
    T_ab = np.linalg.inv(T_a) @ T_b

    print(f"{a} -> {b}")
    print(T_ab)
    print()