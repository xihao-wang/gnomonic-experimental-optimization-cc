# Third-party dependencies for tracker_pipeline

FastReID should be installed here for project-relative imports:

```bash
git clone https://github.com/JDAI-CV/fast-reid.git tracker_pipeline/third_party/fast-reid
```

Alternatively, add it as a submodule if the exact external revision must be
tracked:

```bash
git submodule add https://github.com/JDAI-CV/fast-reid.git tracker_pipeline/third_party/fast-reid
```

The ReID weight file is not stored here. See:

```text
tracker_pipeline/reid/weights/README.md
```
