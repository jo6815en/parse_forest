import os, re, json, shutil
import numpy as np

IMAGES_TXT = "sparse_txt/images.txt"
IMAGE_DIR = "images"
OUT_DIR = "dataset"

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

def frame_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1

def camera_to_world(q, t):
    R = qvec2rotmat(q)
    t = np.array(t)

    T_wc = np.eye(4)
    T_wc[:3, :3] = R.T
    T_wc[:3, 3] = -R.T @ t
    return T_wc

poses = {}

with open(IMAGES_TXT) as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    if line.startswith("#") or line == "":
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
    name = parts[9]

    poses[name] = camera_to_world(q, t)

    i += 2

names = sorted(poses.keys(), key=frame_number)

os.makedirs(f"{OUT_DIR}/images", exist_ok=True)

metadata = {
    "pose_convention": "T_ab maps coordinates from camera/frame a to camera/frame b",
    "scale": "COLMAP scale is arbitrary unless externally calibrated",
    "num_frames": len(names),
    "frames": []
}

for idx, name in enumerate(names):
    new_name = f"{idx:06d}.jpg"

    shutil.copy(
        os.path.join(IMAGE_DIR, name),
        os.path.join(OUT_DIR, "images", new_name)
    )

    metadata["frames"].append({
        "index": idx,
        "original_name": name,
        "dataset_name": new_name,
        "T_world_camera": poses[name].tolist()
    })

relative = []

for idx in range(len(names) - 1):
    a = names[idx]
    b = names[idx + 1]

    T_wa = poses[a]
    T_wb = poses[b]

    T_ab = np.linalg.inv(T_wb) @ T_wa

    relative.append({
        "from_index": idx,
        "to_index": idx + 1,
        "from_image": f"{idx:06d}.jpg",
        "to_image": f"{idx+1:06d}.jpg",
        "original_from": a,
        "original_to": b,
        "T_ab": T_ab.tolist()
    })

with open(f"{OUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(f"{OUT_DIR}/relative_poses.json", "w") as f:
    json.dump(relative, f, indent=2)

print(f"Saved dataset with {len(names)} images")
print(f"Saved {len(relative)} relative poses")
print(f"Output: {OUT_DIR}/")