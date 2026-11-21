
import cv2
import os
import time

try:
    from .matching import classify_match
except ImportError:
    from matching import classify_match


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
FACES_DIR = os.path.join(BASE_DIR, "data", "registered_faces")

YUNET = os.path.join(
    MODEL_DIR, "face_detection_yunet_2023mar.onnx"
)

SFACE = os.path.join(
    MODEL_DIR, "face_recognition_sface_2021dec.onnx"
)

WINDOW_NAME = "Face Recognition System"

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")


# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================

def initialize_models():

    if not os.path.isfile(YUNET):
        raise FileNotFoundError(f"YuNet model not found: {YUNET}")

    if not os.path.isfile(SFACE):
        raise FileNotFoundError(f"SFace model not found: {SFACE}")

    detector = cv2.FaceDetectorYN.create(
        YUNET, "", (320, 320), 0.9, 0.3, 5000
    )

    recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

    print("YuNet detector initialized.")
    print("SFace recognizer initialized.")

    return detector, recognizer


# ==========================================
# 3. REGISTERED FACE LOADING
# ==========================================

def load_registered_faces(detector, recognizer, faces_dir):

    os.makedirs(faces_dir, exist_ok=True)

    registered_faces = []

    for filename in sorted(os.listdir(faces_dir)):

        if not filename.lower().endswith(SUPPORTED_FORMATS):
            continue

        path = os.path.join(faces_dir, filename)

        image = cv2.imread(path)

        if image is None:
            print(f"WARNING: Could not read image: {filename}")
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"WARNING: No face found in: {filename}")
            continue

        # Select the largest face
        face = max(faces, key=lambda f: f[2] * f[3])

        aligned = recognizer.alignCrop(image, face)

        feature = recognizer.feature(aligned)

        if feature is None:
            print(f"WARNING: Feature extraction failed: {filename}")
            continue

        name = os.path.splitext(filename)[0]

        registered_faces.append((name, feature))

        print(f"Registered: {name}")

    print(f"\nTotal registered faces: {len(registered_faces)}")

    return registered_faces


# ==========================================
# 4. FACE MATCHING
# ==========================================

def recognize_face(feature, registered_faces, recognizer):

    best_name = "Unknown"
    best_score = -1.0

    for name, registered_feature in registered_faces:

        score = recognizer.match(
            feature,
            registered_feature,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        if score > best_score:
            best_score = score
            best_name = name

    best_name = classify_match(best_name, best_score)

    return best_name, best_score


# ==========================================
# 5. FACE DISPLAY
# ==========================================

def draw_face_result(frame, face, name, score):

    x, y, fw, fh = map(int, face[:4])

    color = (
        (0, 200, 0)
        if name != "Unknown"
        else (0, 0, 255)
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + fw, y + fh),
        color,
        2
    )

    label = f"{name} ({score:.2f})"

    cv2.putText(
        frame,
        label,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return frame


# ==========================================
# 6. WEBCAM RECOGNITION
# ==========================================

def run_recognition(detector, recognizer, registered_faces):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        cap.release()
        return

    print("\nFace recognition started.")
    print("Press Q or ESC to quit.")

    frame_count = 0
    benchmark_start = time.perf_counter()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Could not read webcam frame.")
                break

            height, width = frame.shape[:2]

            detector.setInputSize((width, height))

            _, faces = detector.detect(frame)

            if faces is not None:

                for face in faces:

                    aligned = recognizer.alignCrop(frame, face)

                    feature = recognizer.feature(aligned)

                    if feature is None:
                        continue

                    name, score = recognize_face(
                        feature,
                        registered_faces,
                        recognizer
                    )

                    frame = draw_face_result(
                        frame,
                        face,
                        name,
                        score
                    )

            frame_count += 1
            elapsed = time.perf_counter() - benchmark_start
            if frame_count % 30 == 0:
                print(f"Average FPS: {frame_count / elapsed:.2f}")

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("Exit key pressed.")
                break

            try:

                if cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE
                ) < 1:

                    print("Window closed.")
                    break

            except cv2.error:
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("Webcam released. Program closed.")


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 45)
    print("       FACE RECOGNITION SYSTEM")
    print("=" * 45)

    detector, recognizer = initialize_models()

    registered_faces = load_registered_faces(
        detector,
        recognizer,
        FACES_DIR
    )

    if not registered_faces:

        print("\nERROR: No registered faces found.")
        print("Add clear face photos to data/registered_faces.")

        return

    run_recognition(
        detector,
        recognizer,
        registered_faces
    )


# ==========================================
# 8. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
