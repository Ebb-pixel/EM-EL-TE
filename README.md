# EM-EL-TE

Absolutely! Below is a **complete `README.md`** that you can copy and paste directly into your GitHub repo.

It reflects your setup using **RTMDet + RTMPose**, and assumes that model weights (`.pth` files) are hosted externally (you can update those links later):

---

````markdown
# 🏊 Underwater Swimmer Pose Estimation using RTMDet + RTMPose

This project implements a **two-stage pose estimation pipeline** for detecting and analyzing swimmers in underwater videos. It uses:

- ✅ **RTMDet** to detect the swimmer (bounding box)
- ✅ **RTMPose** to estimate keypoints (e.g., head, arms, legs) within the detected bounding box

Built using [MMPose](https://github.com/open-mmlab/mmpose) and [MMDetection](https://github.com/open-mmlab/mmdetection), and trained on a **custom underwater dataset**.

---

## 📌 Pipeline Overview

```text
[ Underwater Frame ]
        ↓
[ RTMDet (Swimmer Detection) ]
        ↓
[ Bounding Box ]
        ↓
[ RTMPose (Pose Estimation) ]
        ↓
[ Keypoints + Skeleton Visualization ]
````

---

## 📁 Project Structure

```bash
.
├── configs/
│   ├── rtmdet_underwater/
│   │   └── rtmdet_tiny_1class_underwater.py
│   └── underwater/
│       ├── rtmpose-l_underwater.py
│       └── rtmpose-m_underwater.py
├── custom_underwater_video_pose_inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧩 Dependencies

Install dependencies using OpenMMLab tools.

### Clone and Install MMDetection

```bash
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip install -v -e .
```

### Clone and Install MMPose

```bash
git clone https://github.com/open-mmlab/mmpose
cd mmpose
pip install -v -e .
```

### Install Additional Dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

## 🏋️‍♀️ Training

### RTMDet (Swimmer Detector)

```bash
python tools/train.py configs/rtmdet_underwater/rtmdet_tiny_1class_underwater.py \
    --work-dir work_dirs/rtmdet_tiny_underwater
```

### RTMPose (Pose Estimator)

```bash
python tools/train.py configs/underwater/rtmpose-l_underwater.py \
    --work-dir work_dirs/rtmpose-l_underwater
```

For a lighter model:

```bash
python tools/train.py configs/underwater/rtmpose-m_underwater.py \
    --work-dir work_dirs/rtmpose-m_underwater
```

---

## 🎯 Inference

Run full pipeline inference using both detector and pose estimator:

```bash
python custom_underwater_video_pose_inference.py \
    --det-config configs/rtmdet_underwater/rtmdet_tiny_1class_underwater.py \
    --det-checkpoint path/to/rtmdet.pth \
    --pose-config configs/underwater/rtmpose-l_underwater.py \
    --pose-checkpoint path/to/rtmpose.pth \
    --input path/to/your_video_or_images \
    --output output/dir \
    --smooth
```

### ✅ Features:

* Temporal smoothing with sliding-window average
* PoseLocalVisualizer for keypoint rendering
* Works with both image sequences and video files

---

## 🧪 Evaluation

Evaluate the trained pose estimation model:

```bash
python tools/test.py configs/underwater/rtmpose-l_underwater.py \
    work_dirs/rtmpose-l_underwater/best_AP.pth \
    --eval bbox keypoints
```

---

## 📈 Results

| Model     | AP (Validation) | Notes                  |
| --------- | --------------- | ---------------------- |
| RTMPose-L | \~0.998         | High accuracy, heavier |
| RTMPose-M | TBD             | Lighter, faster        |

---

## 🧠 Dataset Format

* **COCO-style JSON** with:

  * `images`
  * `annotations` (bounding boxes + keypoints)
  * `categories` (including `keypoints` and `skeleton`)
* Empty frames (no annotations) included to improve robustness

---

## 🚫 Checkpoints Not Included

Model weights (`.pth` files) are not tracked in this repo due to GitHub size limits.

### 🔗 Download Pretrained Weights:

* RTMDet Checkpoint: [Download RTMDet](https://your-link.com)
* RTMPose Checkpoint: [Download RTMPose](https://your-link.com)

> Replace the above links with your actual Google Drive / Hugging Face links.

---

## 📦 Optional Streamlit GUI

If you’ve built a `streamlit_app.py`, you can run:

```bash
streamlit run streamlit_app.py
```

> NOTE: The `web_interface/` folder is ignored in Git due to size.

---

## 🧾 .gitignore Highlights

```gitignore
*.pth
*.mp4
*.avi
*.mov
web_interface/
__pycache__/
```

---

## 🙏 Acknowledgments

* [MMPose](https://github.com/open-mmlab/mmpose)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* RTMPose: [RTMPose Paper](https://arxiv.org/abs/2303.07399)

---


