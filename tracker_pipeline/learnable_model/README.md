# Tracker weights

Place learned tracker checkpoints here. Binary checkpoints are ignored by git via
the repository-wide `*.pt` and `*.pth` rules.

Expected temporal model checkpoint for the default StrongSORT runner:

```text
tracker_pipeline/learnable_model/gt_l15_infonce_track.pt
```

This checkpoint is the GT-trained temporal association model used by:

```bash
python tracker_pipeline/process_video_strongsort.py \
  --learned-temporal \
  --fuse-learned-temporal
```
