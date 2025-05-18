# PotWise

# 🔧 Setup Guide

To ensure smooth operation of this project, we recommend creating a clean `conda` virtual environment.

## 1 Create a Conda Environment (Python 3.9)

```bash
conda create -n yoloenv python=3.9
```

## 2 Activate the Environment

```bash
conda activate yoloenv
```

---

## ⚡ (Optional) Enable GPU Acceleration

For faster performance, check your system's CUDA version and install the corresponding PyTorch build from the [official PyTorch website](https://pytorch.org/).

> 💡 If your CUDA version is not listed on the PyTorch site, you can download it directly from the [NVIDIA CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads).  
> Currently supported versions include: **CUDA 11.8, 12.4, and 12.6**

**Note:** If you plan to use CPU only, you can skip this step.

---

## 3 Install Dependencies

Once the environment is active, install the necessary packages:

```bash
pip install ultralytics
```

---

## 4 Data

All data we used can be found in YOLO/data, containing 32 kinds of ingredients in total.

---

✅ That’s it! Your are now ready to run YOLO-based models using the [Ultralytics](https://github.com/ultralytics/ultralytics) library.

# 🚀 Run

To run the bot locally, you can run Telegram Bot/Bot.py

# Example Output

<img src="https://github.com/user-attachments/assets/0dd5b570-51b1-4c37-a9d8-33d14e215fe7" width="400"/>


