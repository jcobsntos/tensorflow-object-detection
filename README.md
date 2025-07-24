
# 🧠 Real-Time Object Detection System using TensorFlow

A real-time object detection system that can identify and track objects in both static images and live video streams using TensorFlow, OpenCV, and COCO API.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.4-orange?style=for-the-badge&logo=tensorflow">
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
<img width="2175" height="2258" alt="Image" src="https://github.com/user-attachments/assets/a325b54d-8076-414c-93c0-2b75b8b4be7c" />
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


## 🧠 Code Structure & Explanation

### 🖼️ Image Detection Program

<details>
<summary><strong>📥 Input 1–3: File Setup & Library Imports</strong></summary>

After setting up the environment, navigate to the `object_detection` directory and create a new Python file. Use PyCharm to import all required libraries.

```bash
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
```
</details>

<details>
<summary><strong>📦 Input 4: Download and Define the Model</strong></summary>

Download a pre-trained model based on the COCO dataset and specify the frozen inference graph path provided by TensorFlow.

```bash
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# This script is more on the outdated side as TensorFlow has updated the object_detection_tutorial.py script.
# I am currently using this code as the new script has errors in it when trying to run on a local PC I haven't worked
# out yet. This code has also been edited to work on local PC as it previously did not. Check last few lines

# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[1]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
```
</details>

<details>
<summary><strong>🧊 Input 5: Extract the Frozen Inference Graph</strong></summary>

Automatically download and extract the TensorFlow model and locate the frozen inference graph used for detection.

```bash
# ## Download Model

# In[2]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[3]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
```
</details>

<details>
<summary><strong>🧾 Input 6: Load Labels and Prepare Image Data</strong></summary>

Load the label map and convert image data into a NumPy array. Set the path for images using a naming pattern like `image1.jpg`, `image2.jpg`, etc.

```bash
# ## Loading label map
# Label maps map indicatces to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[4]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[5]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[6]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 34) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
```
</details>

<details>
<summary><strong>🎯 Input 7: Run Detection and Save Output</strong></summary>

Perform detection on one or more images, draw bounding boxes, label each object, and save the output to a folder named `outputs`.

```bash
# In[7]:

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        i = 0   # add variable for a janky fix
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5,min_score_thresh=0.3)

            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            # plt.show()    # when running on local PC, matplotlib is not set for graphical display so instead we
            # can save the outputs to a folder I named outputs (make sure to add this folder into object_detection)
            plt.savefig("outputs/detection_output{}.png".format(i))
            i = i+1
```
</details>

---

### 🎥 Video & Real-Time Detection Program

<details>
<summary><strong>📥 Input 1–3: Create Python File, Import Libraries & Open Camera</strong></summary>

Use PyCharm to create a new Python file and import all required libraries. Add OpenCV and initialize the webcam for real-time video capture.

```bash
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import cv2

cap = cv2.VideoCapture(r'C:\users\josh\Pictures\Object Detection\Videos\animals.mp4')   # if you have multiple webcams change the value to the correct one

```
</details>

<details>
<summary><strong>📦 Input 4: Load COCO Model</strong></summary>

Just like in the image version, download a COCO-trained model and specify the frozen inference graph path.

```bash
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[1]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

```
</details>

<details>
<summary><strong>🧊 Input 5: Extract Graph for Real-Time Detection</strong></summary>

Ensure the frozen inference graph is available by downloading and extracting if needed.

```bash
# ## Download Model

# In[2]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[3]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
```
</details>

<details>
<summary><strong>🧾 Input 6: Load Labels & Convert Frame Data</strong></summary>

Load class labels and convert captured video frames into NumPy arrays for detection.

```bash
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[4]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[5]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[6]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 6) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

```
</details>

<details>
<summary><strong>📽️ Input 7: Display Video with Detected Objects</strong></summary>

Use OpenCV to create a display window (`object_detection`) and overlay detected object classes with confidence scores. Frames refresh every 25ms.

```bash
# In[7]:

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5,min_score_thresh=0.3)

            cv2.imshow('object detection', cv2.resize(image_np, (1200, 800)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

```
</details>

---

## 🖼️ Sample Outputs

> Object Detection Outputs

### 📸 Image Detection Output

<img width="1200" height="800" alt="Image" src="https://github.com/user-attachments/assets/ed50a110-e5b1-46db-a3ec-f219d41a1339" />
<img width="1200" height="800" alt="Image" src="https://github.com/user-attachments/assets/bed68839-6ade-4e50-82b4-4dd7df1e2389" />
<img width="1200" height="800" alt="Image" src="https://github.com/user-attachments/assets/baee279c-deb7-4854-8f87-adb4a611e29c" />
<img width="1200" height="800" alt="Image" src="https://github.com/user-attachments/assets/467eb3e4-421e-4cd4-b9f4-e326ae73b97a" />

### 🎞️ Video Detection Output

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/487620cb-29de-434f-9da8-3426068bc6c7" />
<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/b5c0b027-ac49-4ed7-bc6f-d0556e761d48" />
<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/fa0428f0-ffc8-4cff-b6e9-6e23e53de4df" />
<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/8cfc863b-8491-4272-a710-5c1a505fc426" />

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
- TensorFlow 
- OpenCV Contributors
- COCO Dataset
- Protobuf Community

---

## 👤 Author

**Jacob Santos**  
GitHub: [@jcobsntos](https://github.com/jcobsntos)

---
