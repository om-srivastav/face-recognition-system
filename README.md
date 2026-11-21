# Face Recognition System

A real-time face recognition system built using **Python, OpenCV, YuNet, and SFace**. The system detects faces through a webcam, extracts facial feature embeddings, and compares them against registered identities using cosine similarity.

## Features

- Real-time face detection using YuNet.
- Facial feature extraction using SFace.
- Face matching using cosine similarity.
- Unknown-person detection using a configurable similarity threshold.
- Multiple registered identity support.
- Modular code structure.
- Automated unit tests.
- Evaluation metrics and threshold analysis.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and computer vision |
| YuNet | Face detection |
| SFace | Face feature extraction |
| NumPy | Numerical operations |
| Matplotlib | Evaluation visualization |
| unittest | Unit testing |

## Project Architecture

```text
face-recognition-system/
│
├── src/
│   ├── detection.py
│   ├── main.py
│   ├── register_face.py
│   ├── recognition.py
│   └── matching.py
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── data/
│   └── registered_faces/
│
├── evaluation/
│   ├── evaluate.py
│   ├── test_images/
│   ├── results.csv
│   └── threshold_analysis.png
│
├── tests/
│   ├── test_matching.py
│   ├── test_gallery.py
│   └── test_no_face.py
│
├── docs/
│   └── threshold_optimization.md
│
├── requirements.txt
└── README.md
```

## How It Works

1. **Face Detection:** YuNet detects faces and their facial landmarks in the input image.
2. **Face Alignment:** SFace aligns the detected face using facial landmarks.
3. **Feature Extraction:** SFace generates a numerical feature representation of the face.
4. **Face Matching:** The extracted feature is compared with registered face features using cosine similarity.
5. **Identity Classification:** The system identifies the best matching identity or labels the face as `Unknown` based on a similarity threshold.

## Installation

Clone the repository:

```bash
git clone https://github.com/om-srivastav/face-recognition-system.git
cd face-recognition-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

Download the required YuNet and SFace ONNX model files and place them inside the `models/` directory.

Required filenames:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

The model files are not included in this repository.

## Registering Faces

Place clear face images in:

```text
data/registered_faces/
```

Use image filenames as identity labels. For example:

```text
data/registered_faces/
    person_name.jpg
```

## Run the System

Launch the webcam-based recognition system:

```bash
python src/recognition.py
```

Press **Q** or **ESC** to exit.

## Face Matching Threshold

The current cosine similarity threshold is:

```python
THRESHOLD = 0.45
```

A similarity score below the threshold is classified as `Unknown`.

The threshold was evaluated across values from 0.30 to 0.70 using the current evaluation dataset. The selected value is a preliminary operating point and may require recalibration with a larger and more diverse dataset.

## Evaluation

Current evaluation results:

| Metric | Result |
|---|---:|
| True Positives | 4 |
| True Negatives | 6 |
| False Positives | 0 |
| False Negatives | 0 |
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| False Acceptance Rate | 0% |
| False Rejection Rate | 0% |

**Dataset limitation:** These results are based on only 10 evaluated images: 4 known and 6 unknown. One image without a detectable face was skipped. These preliminary results do not establish real-world accuracy or generalization.

See `docs/threshold_optimization.md` for threshold analysis.

## Run Tests

Execute the unit test suite:

```bash
python -m unittest discover -s tests -v
```

Current test suite: **7 tests passing**.

## Limitations

- Performance and accuracy depend on image quality, lighting, pose, and camera conditions.
- Evaluation dataset is currently small.
- Recognition performance needs testing across more identities and environmental conditions.
- The current system is a prototype, not a production-grade biometric security solution.

## Future Improvements

- Improve recognition under varying lighting and backgrounds.
- Optimize webcam processing performance.
- Expand the evaluation dataset.
- Add more robust identity registration and management.
- Improve testing and performance benchmarking.

## Author

**Om Srivastav**

GitHub: [om-srivastav](https://github.com/om-srivastav)

---

*Built as a computer vision project to explore face detection, facial feature extraction, similarity-based recognition, and software testing.*
