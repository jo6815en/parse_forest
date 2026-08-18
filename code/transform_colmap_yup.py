import numpy as np
from pathlib import Path

inp = Path("sparse_txt")
out = Path("sparse_yup_txt")
out.mkdir(exist_ok=True)

R_plane = np.array([
    [0.992708325386, 0.071871422231, 0.096771791577],
    [0.071871422231, 0.291594058275, -0.953838288784],
    [-0.096771791577, 0.953838288784, 0.284302324057],
])

R_flip = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
])

R_yup = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
])

A = R_yup @ R_flip @ R_plane

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ])

def rotmat2qvec(R):
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
        [R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]]
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3,0,1,2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q

# copy unchanged files if they exist
for name in ["cameras.txt", "rigs.txt", "frames.txt"]:
    if (inp/name).exists():
        (out/name).write_text((inp/name).read_text())

# transform points3D
with open(inp/"points3D.txt") as f, open(out/"points3D.txt", "w") as g:
    for line in f:
        if line.startswith("#") or not line.strip():
            g.write(line)
            continue
        p = line.split()
        xyz = np.array(list(map(float, p[1:4])))
        xyz2 = A @ xyz
        p[1:4] = [f"{v:.12f}" for v in xyz2]
        g.write(" ".join(p) + "\n")

# transform camera poses
with open(inp/"images.txt") as f, open(out/"images.txt", "w") as g:
    lines = f.readlines()

i = 0
with open(out/"images.txt", "w") as g:
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            g.write(line)
            i += 1
            continue

        p = line.split()
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))

        Rcw_old = qvec2rotmat(q)
        C_old = -Rcw_old.T @ t
        C_new = A @ C_old
        Rcw_new = Rcw_old @ A.T
        t_new = -Rcw_new @ C_new
        q_new = rotmat2qvec(Rcw_new)

        p[1:5] = [f"{v:.12f}" for v in q_new]
        p[5:8] = [f"{v:.12f}" for v in t_new]
        g.write(" ".join(p) + "\n")

        if i + 1 < len(lines):
            g.write(lines[i+1])
        i += 2

print("Skrev transformerad modell till sparse_yup_txt/")
