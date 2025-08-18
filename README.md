# 🏊‍♂️ Underwater Swimmer Pose Estimation  

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)  
![Framework](https://img.shields.io/badge/OpenMMLab-MMPose%20%7C%20MMDetection-green)  
![Torch](https://img.shields.io/badge/PyTorch-2.0-red)  

A two-stage pipeline for **underwater swimmer pose estimation**:  

1. **Detection** – An `RTMDet` model detects the swimmer in each frame.  
2. **Pose Estimation** – The cropped bounding box is passed to an `RTMPose` model to predict keypoints.  

Outputs:  
- 🎥 Annotated video with skeleton overlay.  
- 📄 JSON file containing keypoint coordinates per frame.  

---

## 📂 Repository Structure

rtmpose_annotation/
│
├── training/ # Training configs
│ └── configs/
│ ├── rtmdet_underwater/
│ │ └── rtmdet_tiny_1class_underwater.py
│ └── underwater/
│ ├── rtmpose-l_underwater.py
│ └── rtmpose-m_underwater.py
│
├── inference/ # Inference scripts
│ ├── configs/ # Flattened configs for standalone inference
│ │ ├── rtmdet_tiny_infer_flat.py
│ │ └── rtmpose-l_infer_flat.py
│ ├── custom_inference.py # CLI inference
│ ├── gui_inference.py # GUI inference (file chooser + progress)
│ └── tools/
│ └── prepare_inference_configs.py
│
├── Dataset_split_script/ # Dataset preparation helpers
│ ├── split_coco_person_detection.py
│ └── split_data.py
│
└── .gitignore


---

## ⚙️ Installation

```bash
# 1. Create conda env
conda create -n mmpose python=3.9 -y
conda activate mmpose

# 2. Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Clone and install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
cd ..

# 4. Clone and install MMPose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
cd ..

# 5. Extra requirements
pip install -r requirements.txt

📌 Config Inheritance ⚠️

The training configs here (e.g. rtmpose-l_underwater.py) inherit from _base_ configs inside the official repos (mmpose/configs and mmdetection/configs).

That means:

✅ You MUST clone MMPose and MMDetection repos.

❌ If you only copy configs without repos, training will fail with:

FileNotFoundError: Cannot find config file: ../_base_/models/...

🚀 Training
1️⃣ Train RTMDet (Swimmer Detector)

python tools/train.py training/configs/rtmdet_underwater/rtmdet_tiny_1class_underwater.py

2️⃣ Train RTMPose (Pose Estimator)

python tools/train.py training/configs/underwater/rtmpose-l_underwater.py

Checkpoints and logs → work_dirs/.

🎯 Inference
🔹 CLI Inference

python inference/custom_inference.py --input path/to/video.mp4 --device cuda

Outputs:

output/result.mp4 – annotated video

output/keypoints.json – keypoint coordinates

🔹 GUI Inference

For end-users (double-click .exe or run Python):

File chooser opens to select video.

Progress bar shows during processing.

Notification when inference is done.

python inference/gui_inference.py

📊 Dataset Preparation

Use provided scripts:

# Split into train / val / test
python Dataset_split_script/split_data.py

# Filter COCO person annotations
python Dataset_split_script/split_coco_person_detection.py

📝 Notes

Detector: RTMDet-Tiny (1-class swimmer).

Pose Estimator: RTMPose-L (custom-trained).

Outputs: video + JSON.

For client delivery: packaged as PyInstaller .exe (gui_inference.py).

🙌 Credits

MMPose

MMDetection

Dataset collected & annotated for this project - Aquatics GB