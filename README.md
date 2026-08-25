# Parse Forest pipeline update

The pipeline is resumable. Each run lives under:

```text
datasets/<dataset>/reconstructions/YYYYMMDD_HHMMSS/
```

## Install

The dense point-cloud transformer needs `plyfile`:

```bash
pip install plyfile
```

## Start a new run

From the repository root:

```bash
python code/run_pipeline.py --dataset torpa --video video3.mp4 --fps 3
```

The video path is relative to `datasets/<dataset>/` unless an absolute path is supplied.

## Resume an existing run

Resume the latest run for a dataset:

```bash
python code/run_pipeline.py --dataset torpa --resume
```

Resume a specific run:

```bash
python code/run_pipeline.py --dataset torpa --resume 20260824_142747
```

When resuming, `--video` is not required if the frames already exist. If a run has no frames, provide the video path as well:

```bash
python code/run_pipeline.py --dataset torpa --video video3.mp4 --resume 20260824_142747
```

## What resume does

The pipeline checks the filesystem/database before every step and skips work that is already complete:

1. frames
2. COLMAP feature extraction
3. sequential matching
4. sparse mapping
5. sparse TXT conversion
6. image undistortion
7. PatchMatch Stereo
8. stereo fusion
9. CloudCompare transformation matrix
10. Y-up sparse model
11. Y-up dense point cloud
12. camera pose export
13. camera trajectory export

If a run was stopped after sparse reconstruction, `--resume` continues from there. If dense is already complete, it skips dense. If the transformation and Y-up outputs already exist, those are skipped too.

The script does not delete or overwrite completed run data when resuming.

## CloudCompare step

The only intentionally manual step is choosing the world alignment in CloudCompare. After dense reconstruction, the script asks for the path to the saved 4x4 matrix. The matrix is copied into the run as:

```text
transformation.txt
```

The same final transformation is then applied to the sparse model, dense point cloud, and exported camera poses.

## Output

A completed run looks like:

```text
datasets/torpa/reconstructions/20260824_142747/
├── images/
├── transformation.txt
├── colmap/
│   ├── database.db
│   ├── sparse/
│   ├── sparse_txt/
│   ├── sparse_yup_txt/
│   └── dense/
│       ├── fused.ply
│       └── fused_yup.ply
└── reconstruction/
    ├── images/
    ├── metadata.json
    ├── camera_poses.json
    ├── relative_poses.json
    └── camera_trajectory.ply
```
