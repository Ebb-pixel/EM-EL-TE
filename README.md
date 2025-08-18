# 🏊‍♂️ Underwater Swimmer Pose Estimation  

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)  
![Framework](https://img.shields.io/badge/OpenMMLab-MMPose%20%7C%20MMDetection-green)  
![Torch](https://img.shields.io/badge/PyTorch-2.0-red) 

This project implements a **two-stage pipeline** for underwater swimmer analysis using **RTMDet** (detection) and **RTMPose** (pose estimation).  

1. **Detection (RTMDet):** Locates the swimmer in each frame.  
2. **Pose Estimation (RTMPose):** Predicts keypoints from the detected bounding box.  
3. **Output:** Annotated video + JSON file with keypoints for downstream analysis.  

---

## 📂 Repository Structure  

```

rtmpose\_annotation/
│── training/              # Training configs for RTMDet & RTMPose
│   └── configs/
│       ├── rtmdet\_underwater/
│       │   └── rtmdet\_tiny\_1class\_underwater.py
│       └── underwater/
│           ├── rtmpose-l\_underwater.py
│           └── rtmpose-m\_underwater.py
│
│── inference/             # Inference scripts & configs
│   ├── custom\_inference.py
│   ├── gui\_inference.py
│   ├── configs/
│   │   ├── rtmdet\_tiny\_infer\_flat.py
│   │   └── rtmpose-l\_infer\_flat.py
│   └── tools/
│       └── prepare\_inference\_configs.py
│
│── Dataset\_split\_script/  # Dataset preparation utilities
│   ├── split\_coco\_person\_detection.py
│   └── split\_data.py

````

---

## ⚙️ Installation  

We use **MMPose** and **MMDetection** as the backbone.  
Clone this repo along with official MMPose and MMDetection to ensure configs work properly.  

```bash
# Create conda env
conda create -n mmpose python=3.9 -y
conda activate mmpose

# Install PyTorch (check CUDA version at pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install mmcv
pip install -U openmim
mim install mmcv==2.0.1

# Clone MMPose & MMDetection
git clone https://github.com/open-mmlab/mmpose.git
git clone https://github.com/open-mmlab/mmdetection.git

# Install both
cd mmpose
pip install -e .
cd ../mmdetection
pip install -e .
````

---

## 🔗 Download Pretrained Weights & Executable  

- RTMPose-L Underwater model weights:
- [![Download RTMPose-L](https://img.shields.io/badge/Download-RTMPose--L-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/uc?export=download&id=1mQsvfwqmI_VVCbJjAS8RbAHE-z94kTEz)  
- RTMDet Tiny Underwater detector weights:
- [![Download RTMDet](https://img.shields.io/badge/Download-RTMDet-green?style=for-the-badge&logo=google-drive)](https://drive.google.com/uc?export=download&id=16XoZciKUtFzZoXHsfgBdtinbc0H47dzc)  
- Standalone Windows executable:
- [![Download .exe](https://img.shields.io/badge/Download-Windows--Executable-orange?style=for-the-badge&logo=windows)](https://drive.google.com/uc?export=download&id=1vimRGwq2KG98vJqaeZjFHs2gKCD7y1FN)
- CLI executable:
- [![Download CLI .exe](https://img.shields.io/badge/Download-CLI--Executable-red?style=for-the-badge&logo=terminal)](https://drive.google.com/uc?export=download&id=1CkEfsQmZG41cTi7Y5JdzUOvDVf2UVfzs)


## 🚀 Training

Run training for **detection** or **pose estimation**.

```bash
# Example: Train RTMDet on underwater dataset
python tools/train.py training/configs/rtmdet_underwater/rtmdet_tiny_1class_underwater.py

# Example: Train RTMPose on underwater dataset
python tools/train.py training/configs/underwater/rtmpose-l_underwater.py
```

> ⚠️ **Important**: Training configs inherit from `_base_` configs inside MMPose & MMDetection.
> That’s why cloning those repos is required.

---

## 🎯 Inference

We provide **two inference options**:

### 1. CLI Inference

```bash
python inference/custom_inference.py --input path/to/video.mp4 --device cuda
```

Arguments:

* `--input` : path to input video
* `--device` : `cpu` or `cuda`

Outputs:

* Annotated video (`output_with_keypoints.mp4`)
* JSON file (`keypoints.json`)

---

### 2. GUI Inference

Run:

```bash
python inference/gui_inference.py
```

This opens a file chooser to select your video.
Progress is displayed in a GUI window, and a message notifies when inference is complete.

---

## 📊 Dataset Preparation

Use our helper scripts:

```bash
# Split dataset into train/val
python Dataset_split_script/split_data.py

# Convert COCO dataset for person detection
python Dataset_split_script/split_coco_person_detection.py
```

---

## 📈 Example Output JSON

Each frame stores the detected keypoints:

```json
{
  "frame_id": 12,
  "keypoints": [
    [320.5, 140.2], 
    [330.1, 200.7], 
    [310.9, 250.3]
  ]
}
```

Where each entry is `[x, y]`.

---

## 🔗 Pipeline Overview

```
Video Input
     │
     ▼
 [ RTMDet ]  --->  Bounding Box
     │
     ▼
 [ RTMPose ] --->  Keypoints
     │
     ├──▶ Annotated Video
     └──▶ JSON Keypoints
```

---

## 🙏 Credits

* [MMPose](https://github.com/open-mmlab/mmpose)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* OpenMMLab team for the toolkits
* Dataset acquired from Aquatics GB
