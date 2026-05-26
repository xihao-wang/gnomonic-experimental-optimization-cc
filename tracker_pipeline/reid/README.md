# ReID integration

This folder contains the project-owned ReID integration layer.

## Runtime layout

Expected default paths from the project root:

```text
tracker_pipeline/third_party/fast-reid/         # FastReID source checkout
tracker_pipeline/reid/configs/bagtricks_S50.yml # project-owned config copy
tracker_pipeline/reid/weights/duke_bot_S50.pth  # local model weights, ignored by git
```

## FastReID source

Do not copy a personal absolute path into the code. Install FastReID under
`tracker_pipeline/third_party/fast-reid`:

```bash
mkdir -p tracker_pipeline/third_party
git clone https://github.com/JDAI-CV/fast-reid.git tracker_pipeline/third_party/fast-reid
```

If the repository should track the exact FastReID revision, use a git submodule
instead:

```bash
git submodule add https://github.com/JDAI-CV/fast-reid.git tracker_pipeline/third_party/fast-reid
```

## Weights

Place the BoT/FastReID weights at:

```text
tracker_pipeline/reid/weights/duke_bot_S50.pth
```

The `.pth` file is intentionally ignored by git. Use Git LFS or an internal
model registry if the company wants to version the weights.
