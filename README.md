# PotWise

# 🔧 Environment Setup Guide

To ensure smooth operation of this project, we recommend creating a clean `conda` virtual environment.

## 1️⃣ Create a Conda Environment (Python 3.9)

```bash
conda create -n yoloenv python=3.9
```

## 2️⃣ Activate the Environment

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

## 3️⃣ Install Dependencies

Once the environment is active, install the necessary packages:

```bash
pip install ultralytics
```

---

✅ That’s it! Your environment is now ready to run YOLO-based models using the [Ultralytics](https://github.com/ultralytics/ultralytics) library.

For more usage instructions, refer to the official documentation or continue below in this repository.
