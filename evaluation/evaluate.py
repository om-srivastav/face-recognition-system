
import cv2
import os
import csv

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
FACES_DIR = os.path.join(BASE_DIR, "data", "registered_faces")
TEST_DIR = os.path.join(BASE_DIR, "evaluation", "test_images")
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation", "results.csv")

YUNET = os.path.join(
    MODEL_DIR, "face_detection_yunet_2023mar.onnx"
)
SFACE = os.path.join(
    MODEL_DIR, "face_recognition_sface_2021dec.onnx"
)

THRESHOLD = 0.45
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Check required files
for path in (YUNET, SFACE):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")

# Initialize models
detector = cv2.FaceDetectorYN.create(
    YUNET, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")


def load_face(image_path):
    """Detect and extract the largest face from an image."""
    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned)


def get_images(folder):
    """Return supported image files recursively."""
    paths = []

    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


# Load reference faces from registered_faces
gallery = []

for filename in os.listdir(FACES_DIR):
    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue

    path = os.path.join(FACES_DIR, filename)
    feature = load_face(path)

    if feature is None:
        print(f"Skipping invalid reference image: {filename}")
        continue

    person_name = os.path.splitext(filename)[0]
    gallery.append((person_name, feature))
    print(f"Loaded reference: {person_name}")

if not gallery:
    raise RuntimeError(
        "No valid registered faces found in data/registered_faces."
    )


def predict(feature):
    """Return the best identity and its cosine similarity score."""
    best_name = "Unknown"
    best_score = -1.0

    for name, reference_feature in gallery:
        score = float(recognizer.match(
            feature,
            reference_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        ))

        if score > best_score:
            best_name = name
            best_score = score

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


# Evaluate known and unknown images
results = []

known_correct = 0
known_rejected = 0
known_misidentified = 0
unknown_correct = 0
unknown_accepted = 0
skipped = 0

known_root = os.path.join(TEST_DIR, "known")
unknown_root = os.path.join(TEST_DIR, "unknown")

# Known images: expected identity comes from folder name
if os.path.isdir(known_root):
    for person_name in sorted(os.listdir(known_root)):
        person_folder = os.path.join(known_root, person_name)

        if not os.path.isdir(person_folder):
            continue

        for image_path in get_images(person_folder):
            feature = load_face(image_path)

            if feature is None:
                print(f"SKIPPED (no face): {image_path}")
                skipped += 1
                continue

            predicted, score = predict(feature)
            expected = person_name

            if predicted == expected:
                status = "Correct"
                known_correct += 1
            elif predicted == "Unknown":
                status = "False Rejection"
                known_rejected += 1
            else:
                status = "Wrong Identity"
                known_misidentified += 1

            results.append([
                "Known", image_path, expected,
                predicted, round(score, 4), status
            ])

            print(
                f"[KNOWN] Expected={expected}, "
                f"Predicted={predicted}, Score={score:.3f}, "
                f"Result={status}"
            )
else:
    print(f"Missing folder: {known_root}")

# Unknown images: none should match a registered identity
if os.path.isdir(unknown_root):
    for image_path in get_images(unknown_root):
        feature = load_face(image_path)

        if feature is None:
            print(f"SKIPPED (no face): {image_path}")
            skipped += 1
            continue

        predicted, score = predict(feature)

        if predicted == "Unknown":
            status = "Correct Rejection"
            unknown_correct += 1
        else:
            status = "False Acceptance"
            unknown_accepted += 1

        results.append([
            "Unknown", image_path, "Unknown",
            predicted, round(score, 4), status
        ])

        print(
            f"[UNKNOWN] Predicted={predicted}, "
            f"Score={score:.3f}, Result={status}"
        )
else:
    print(f"Missing folder: {unknown_root}")

# Save detailed results
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Category", "Image", "Expected",
        "Predicted", "CosineSimilarity", "Result"
    ])
    writer.writerows(results)

known_total = known_correct + known_rejected + known_misidentified
unknown_total = unknown_correct + unknown_accepted
evaluated_total = known_total + unknown_total

# Binary known-vs-unknown metrics for the current single-identity gallery.
# Wrong-identity predictions are reported separately.
tp = known_correct
fn = known_rejected
fp = unknown_accepted
tn = unknown_correct

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0

accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
precision = safe_divide(tp, tp + fp)
recall = safe_divide(tp, tp + fn)
f1_score = safe_divide(2 * precision * recall, precision + recall)
far = safe_divide(fp, fp + tn)
frr = safe_divide(fn, fn + tp)

print("\n========== EVALUATION SUMMARY ==========")
print(f"Threshold: {THRESHOLD}")
print(f"Registered identities: {len(gallery)}")
print(f"Known test images evaluated: {known_total}")
print(f"Unknown test images evaluated: {unknown_total}")
print(f"Images skipped: {skipped}")
print(f"Wrong identity predictions: {known_misidentified}")

if evaluated_total:
    print(f"Overall exact-decision accuracy: {safe_divide(known_correct + unknown_correct, evaluated_total):.2%}")
else:
    print("No test images were evaluated.")

print("\n========== CONFUSION MATRIX ==========")
print("                     Predicted Known   Predicted Unknown")
print(f"Actual Known              {tp:>5}                {fn:>5}")
print(f"Actual Unknown            {fp:>5}                {tn:>5}")

print("\n========== CLASSIFICATION METRICS ==========")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1_score:.2%}")
print(f"FAR:       {far:.2%}")
print(f"FRR:       {frr:.2%}")

if known_total:
    print(f"Known correct identification rate: {safe_divide(known_correct, known_total):.2%}")
    print(f"Known false rejection rate: {safe_divide(known_rejected, known_total):.2%}")
    print(f"Known wrong identity rate: {safe_divide(known_misidentified, known_total):.2%}")

if unknown_total:
    print(f"Unknown correct rejection rate: {safe_divide(unknown_correct, unknown_total):.2%}")
    print(f"Unknown false acceptance rate: {safe_divide(unknown_accepted, unknown_total):.2%}")

print(f"\nDetailed results saved to: {OUTPUT_CSV}")
