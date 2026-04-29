# RailCalibNet

Code for the paper: **A LiDAR-Camera Calibration Network for Dynamic Railway Scenes Based on Hybrid Attention**.



## 1. lidar_pro

`lidar_pro` uses CUDA for point cloud projection and is much faster than CPU projection during training.  
Please install it in your training environment:

```bash
cd lidar_pro_package
pip install -e . --no-build-isolation
```

Notes:

- `-e` installs in editable mode, so you can modify the source without rebuilding a full package each time.
- `--no-build-isolation` reuses `torch` from your current environment during build, which helps reduce version mismatch issues.
- The point cloud data used in this project is already aligned to the camera coordinate system, so it can be projected directly.

Quick verification after installation:

```bash
python -c "import torch;import lidar_pro; print('lidar_pro ok:', hasattr(lidar_pro, 'lidar_pro'))"
```



## 2. Training and Evaluation

Training example:

```bash
python train.py --checkpoints checkpoints
```

Evaluation example:

```bash
python eval.py --checkpoints /path/to/checkpoint.tar
```

## 3. Acknowledgement

Parts of this repository were adapted from the following excellent open-source projects.  
Many thanks to the original authors:

- https://github.com/LvXudong-HIT/LCCNet
- https://github.com/AlexWang0214/MRCNet

If you have any questions or run into issues, feel free to open an issue for discussion.
