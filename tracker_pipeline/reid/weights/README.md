# ReID weights

Download the FastReID/BoT checkpoint from Google Drive:

```text
https://drive.google.com/file/d/1VZQHwOUcwhe8vRA8GNQ6HKzDA8RWWjOI/view?usp=sharing
```

Expected local path from the project root:

```text
tracker_pipeline/reid/weights/duke_bot_S50.pth
```

The binary checkpoint is ignored by git via the repository-wide `*.pth` rule.
Do not commit it to normal Git. If the company needs full reproducibility, store
the file with Git LFS or an internal artifact/model registry instead.
