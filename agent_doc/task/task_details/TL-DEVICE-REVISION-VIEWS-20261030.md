# TL-DEVICE-REVISION-VIEWS-20261030

Main8ff7e78574a0dd8246ac1a5b677bd625939d4dc8 sliced valid-length tensors lose the producer's Python revision attribute. Include storage-base revision in geometry-cache lookup/store; preserve all existing cache guards. See ../../results/view-revision-20261030/README.md for baseline failures, host validation, native commands and branch coverage. Native Torch/CUDA acceptance pending; no GPU performance claim. Deliver via isolated draft PR; maintainers merge.
