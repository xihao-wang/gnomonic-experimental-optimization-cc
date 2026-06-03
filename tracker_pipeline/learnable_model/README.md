# Learned Temporal Model Checkpoints

Binary checkpoints are ignored by git through the repository-wide `*.pt` and
`*.pth` rules.

The StrongSORT runner can load one of these checkpoints with:

```bash
python tracker_pipeline/process_video_strongsort.py \
  --learned-temporal \
  --fuse-learned-temporal \
  --tracker-model tracker_pipeline/learnable_model/<checkpoint>.pt
```

PIROPO database resource:

```text
https://sites.google.com/site/piropodatabase/
```

## Checkpoint Summary

| Checkpoint | Training scenes | Validation scenes | Pair data source | Notes |
|---|---|---|---|---|
| `gt_l15_infonce_track.pt` | PIROPO / Omni3A GT temporal pairs. Exact split was from the early PIROPO experiment. | PIROPO / Omni3A validation split. | `tracker_pipeline/data/temporal_pairs_piropo_l15/` | Early PIROPO temporal association model. |
| `wepdtof_gt_l15_infonce_track_small.pt` | `tech_store`, `jewelry_store`, `jewelry_store_2` | `warehouse` | Old non-dedup WEPDTOF pair datasets. | Early WEPDTOF model trained before hard-negative sharding and before dedup detection. |
| `wepdtof_hardneg5_jtp_valwh.pt` | `jewelry_store`, `jewelry_store_2`, `tech_store`, `printing_store` | `warehouse` | Old detection WEPDTOF hard-negative shard datasets. | Previous hardneg5 model. Used as `old_model` in result folders. |
| `wepdtof_dedup_hardneg5_jtp_valkindergarten.pt` | `jewelry_store`, `jewelry_store_2`, `tech_store`, `printing_store` | `kindergarten` | `tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/` | Latest model. Trained from dedup detection outputs with hard negative sharding. Used as `new_model` in result folders. |

## Latest Training Data Layout

The latest model uses:

```text
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/
```

Training:

```text
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/jewelry_store/*.npz
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/jewelry_store_2/*.npz
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/tech_store/*.npz
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/printing_store/*.npz
```

Validation:

```text
tracker_pipeline/data/temporal_pairs_wepdtof_dedup_l15_hardneg5_shards/kindergarten/*.npz
```

Held-out evaluation scenes used after training:

```text
empty_store
it_office
warehouse
convenience_store
```

## Result Folder Mapping

Current result folders use these model labels:

```text
tracker_pipeline/results/old_det/old_model/
tracker_pipeline/results/dedup/old_model/
```

use:

```text
wepdtof_hardneg5_jtp_valwh.pt
```

and:

```text
tracker_pipeline/results/dedup/new_model/
```

uses:

```text
wepdtof_dedup_hardneg5_jtp_valkindergarten.pt
```
