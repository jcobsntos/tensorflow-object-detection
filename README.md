
# 🧠 Real-Time Object Detection System using TensorFlow

A real-time object detection system that can identify and track objects in both static images and live video streams using TensorFlow, OpenCV, and COCO API.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.4-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

---

## 🎥 Demo Video

📺 Want to see how it works before diving in?  
👉 [Watch the full walkthrough here](https://drive.google.com/file/d/1vBUIjv5vtACw-fuPCQkIH2WTwHPziWDN/view?usp=sharing)

---

## 🚀 Project Highlights

- 🖼️ Detect objects in images and real-time video using webcam or video files.
- ⚡ Powered by TensorFlow Object Detection API and pre-trained models.
- 🎯 Draws accurate bounding boxes and classifies objects.
- 🧪 Trial and error tested in PyCharm and Jupyter Notebook.
- 🔧 Integrates with COCO API for custom dataset handling.
- ✅ Saves processed outputs for later analysis.

---

## 🧩 Dependencies

TensorFlow Object Detection API relies on the following libraries:

- 📦 **Protobuf 2.6**
- 🖼️ **Pillow 1.0**
- 📄 **lxml**
- 🧠 **tf-slim**
- 📒 **Jupyter Notebook**
- 📊 **Matplotlib**
- 🔶 **TensorFlow**
- ⚙️ **Cython**
- 🔄 **contextlib2**
- 🐒 **COCO API**
- 🧮 **NumPy**
- 📈 **scikit-learn**

---

## ⚙️ Python Environment Setup

> It's recommended to use a virtual environment to isolate dependencies and avoid conflicts.

### 🐍 Step-by-step:

```bash
# Create a virtual environment (Anaconda recommended)
conda create -n object_detection_env python=3.7
conda activate object_detection_env

# Upgrade pip
pip install --upgrade pip
```

---

## 🛠️ Installation Guide

<details open>
<summary><strong>📥 Step 1: Install Required Applications</strong></summary>

- [Anaconda (Python 3.7)](https://www.anaconda.com/products/individual#windows)
- [Git](https://git-scm.com/downloads)
- [Python 3.7](https://www.python.org/downloads/)
- [PyCharm](https://www.jetbrains.com/pycharm/download/)
- [OpenCV 4.5.1](https://sourceforge.net/projects/opencvlibrary/)
- [TensorFlow](https://pypi.org/project/tensorflow/)

Install TensorFlow 2.4.1:
```bash
pip install tensorflow==2.4.1
```

Clone the TensorFlow Models:
```bash
git clone https://github.com/tensorflow/models.git
```

</details>

<details>
<summary><strong>📚 Step 2: Install Dependencies</strong></summary>

```bash
pip install --user Cython contextlib2 pillow lxml jupyter matplotlib
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install pycocotools
cp object_detection/packages/tf2/setup.py .
python -m pip install .
python object_detection/builders/model_builder_tf2_test.py
```

</details>

<details>
<summary><strong>🐒 Step 3: Install COCO API (Manual Option)</strong></summary>

1. Download from: https://github.com/cocodataset/cocoapi  
2. Extract and move:
   ```bash
   cocoapi/PythonAPI/pycocotools/ → tensorflow/models/research/
   ```

Or clone via Git:
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
cp -r pycocotools <path_to>/models/research/
```

</details>

<details>
<summary><strong>📓 Step 4: Run Jupyter Notebook Demo</strong></summary>

```bash
python setup.py build
jupyter notebook
```

Then open: `object_detection_tutorial.ipynb`

</details>

<details>
<summary><strong>🧪 Step 5: Run the Object Detection System</strong></summary>

- `detect_image.py` – for static images
- `detect_video.py` – for real-time video feed

Example (image):
```bash
python detect_image.py --image_path path/to/image.jpg
```

Example (video/live):
```bash
python detect_video.py
```

</details>

<details>
<summary><strong>🎥 Step 6: Install OpenCV</strong></summary>

```bash
pip install opencv-python
```

Or download from: https://opencv.org/releases/

</details>

---

## 🖼️ Sample Outputs

> Replace the below links or images with your actual outputs

### 📸 Image Detection Output
![sample-image](path/to/sample-image.jpg)

### 🎞️ Video Detection Output
![sample-video](path/to/sample-video.gif)

---

## 📁 Output Directory

All detection results are saved in:

```
models/object_detection/output/
```

Contents:
- ✅ Annotated images/videos
- ✅ Inference logs
- ✅ Bounding box overlays

---

## 🧰 Tools & Frameworks Used

| Tool         | Version       |
|--------------|----------------|
| Python       | 3.7 (64-bit)   |
| TensorFlow   | 2.4.1 & 1.15   |
| OpenCV       | 4.5.1          |
| PyCharm      | 2020.3.3       |
| Anaconda     | 3 (Python 3.7) |
| Jupyter      | latest         |

---

## 🙌 Acknowledgments

Special thanks to:
- TensorFlow Object Detection Team
- OpenCV Contributors
- COCO Dataset Team
- Protobuf Community

---

## 👤 Author

**Jacob Santos**  
GitHub: [@jcobsntos](https://github.com/jcobsntos)

---
