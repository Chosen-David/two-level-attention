# TL-EARLY-DENSE-20261029

Move the existing short-context dense dispatch ahead of discarded mask/head-map preparation. Main and PR10 source blob match. 192 host-dispatch pairs passed in normal/-O; negative cutoff control rejected. Native Torch/CUDA gates and performance pending; no GPU speed claim. Evidence and remote commands: ../../results/early-dense-dispatch-20261029/README.md. Delivered through PR10; existing unrelated integration conflicts remain.
