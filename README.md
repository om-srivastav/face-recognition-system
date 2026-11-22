# Face Recognition System

A modular face recognition system built with **Python, OpenCV, YuNet, and SFace**. It detects faces, extracts facial feature embeddings, and identifies registered individuals using cosine similarity with an unknown-person rejection mechanism.

The project includes automated tests, threshold analysis, an identity-disjoint evaluation on the Labeled Faces in the Wild (LFW) dataset, and offline performance benchmarking.

## Features

- Face detection using OpenCV YuNet.
- Facial landmark detection and face alignment.
- Feature extraction using OpenCV SFace.
- Face matching using cosine similarity.
- Unknown-person rejection using a configurable threshold.
- Multiple registered identity support.
- Modular recognition pipeline.
- Error handling for missing models and invalid images.
- Automated unit tests.
- LFW-based open-set evaluation.
- Threshold comparison and performance benchmarking.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| OpenCV | Computer vision and face processing |
| YuNet | Face detection |
| SFace | Face feature extraction |
| NumPy | Numerical operations |
| Matplotlib | Evaluation visualization |
| unittest | Automated testing |

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
│   ├── evaluate_lfw.py
│   ├── benchmark.py
│   ├── results.csv
│   └── public_dataset/
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

Generated evaluation outputs and personal biometric data are kept out of version control.

## How It Works

1. **Face Detection:** YuNet detects faces and facial landmarks in an input image.
2. **Face Alignment:** The detected face is aligned using facial landmarks.
3. **Feature Extraction:** SFace generates a numerical feature representation (embedding).
4. **Face Matching:** The embedding is compared against registered face embeddings using cosine similarity.
5. **Identity Classification:** The best matching identity is returned if its similarity score meets the configured threshold; otherwise, the face is classified as `Unknown`.

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
└── person_name.jpg
```

The registration pipeline uses these images to create the registered face gallery.

## Running the System

Launch the webcam-based recognition system:

```bash
python src/recognition.py
```

Press **Q** or **ESC** to exit.

## Face Matching and Threshold

The current configured cosine similarity threshold is:

```python
THRESHOLD = 0.45
```

A similarity score below the threshold is classified as `Unknown`.

The threshold controls the trade-off between false acceptance and false rejection. A lower threshold can accept more genuine matches but may also accept unknown individuals. A higher threshold is more restrictive and may reject genuine matches.

The value of `0.45` is a provisional operating point, not a universal optimum. Its suitability depends on the dataset and operating conditions.

## Evaluation

### 1. Initial Custom Evaluation

The initial small evaluation used:

- 4 known face images.
- 6 unknown face images.
- 1 image without a detectable face was skipped.

The initial experiment achieved 100% accuracy on its 10 evaluated images. This is a small preliminary result and should not be interpreted as evidence of real-world accuracy.

### 2. LFW Identity-Disjoint Evaluation

The system was evaluated using the **Labeled Faces in the Wild (LFW)** dataset.

Dataset information:

- 5,749 identities.
- 13,233 images.
- 100 enrolled identities in the second experiment.
- 100 unknown identities.
- 219 evaluated known probes.
- 100 evaluated unknown probes.

The second experiment excluded 200 identities used in the first experiment through an exclusion manifest. This provides an identity-disjoint split between the two experiments.

**Results at a cosine similarity threshold of 0.45:**

| Metric | Result |
|---|---:|
| Binary Acceptance Accuracy | 96.55% |
| Precision | 100.00% |
| Recall | 94.98% |
| F1-score | 97.42% |
| False Acceptance Rate (FAR) | 0.00% |
| False Rejection Rate (FRR) | 5.02% |
| Exact Known Identification Rate | 94.52% |

Additional observations:

- 207 known probes were correctly identified.
- 1 known probe was assigned the wrong identity.
- 11 known probes were rejected.
- No unknown probes were accepted.

**Evaluation protocol note:** This is a custom open-set gallery/probe identification experiment. It is not the official LFW 6,000-pair face verification benchmark. These results should not be compared directly with published official LFW benchmark scores.

### 3. Offline Performance Benchmark

The processing pipeline was benchmarked using a deterministic sample of 200 LFW images, with 10 warm-up images.

| Measurement | Result |
|---|---:|
| Images processed | 200 |
| Readable images | 200 |
| Images with extracted features | 197 |
| Images without extracted features | 3 |
| Total benchmark time | 3.2005 seconds |
| Overall throughput | 62.49 images/second |
| Average image reading time | 1.18 ms |
| Average face detection time | 3.19 ms |
| Average alignment time | 0.23 ms |
| Average feature extraction time | 11.37 ms |
| Average end-to-end time | 15.83 ms |

These are offline image-processing measurements on the development machine. They are not webcam FPS figures and may vary with hardware and software configuration.

## Automated Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The current test suite contains **7 passing tests**, covering matching logic, gallery handling, and no-face scenarios.

## Evaluation Scripts

Run the initial custom evaluation:

```bash
python evaluation/evaluate.py
```

Run the LFW evaluation:

```bash
python evaluation/evaluate_lfw.py
```

Run the identity-disjoint LFW experiment:

```bash
python evaluation/evaluate_lfw.py --enrolled 100 --unknown 100 --max-probes 5 --seed 20261121 --exclude-manifest evaluation/lfw_results/lfw_split_manifest.csv --output-dir evaluation/lfw_results_disjoint
```

Run the offline performance benchmark:

```bash
python evaluation/benchmark.py --limit 200 --seed 42 --warmup 10
```

## Limitations

- Recognition performance depends on image quality, lighting, pose, and facial visibility.
- The evaluation uses a custom gallery/probe protocol rather than the official LFW verification protocol.
- The identity-disjoint experiment is still limited in scale.
- Threshold selection requires further validation on independent datasets and real-world conditions.
- Offline throughput does not represent webcam performance.
- This project is an educational prototype, not a production-grade biometric security system.

## Future Improvements

- Evaluate on larger and more diverse datasets.
- Validate thresholds using independent held-out data.
- Improve recognition under varying lighting and facial poses.
- Add more robust identity registration and management.
- Explore real-world deployment and performance optimization.

## Author

**Om Srivastav**

GitHub: [om-srivastav](https://github.com/om-srivastav)

---

*Built as a computer vision project exploring face detection, facial feature extraction, similarity-based recognition, evaluation methodology, and software testing.*
