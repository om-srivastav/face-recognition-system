import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "evaluation" / "public_dataset" / "lfw"
MODEL_DIR = BASE_DIR / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
OUTPUT_DIR = BASE_DIR / "evaluation" / "performance_results"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_image(path, detector, recognizer):
    timings = {
        "image_read_ms": 0.0,
        "detection_ms": 0.0,
        "alignment_ms": 0.0,
        "feature_ms": 0.0,
        "total_ms": 0.0,
    }

    total_start = time.perf_counter()

    start = time.perf_counter()
    image = cv2.imread(str(path))
    timings["image_read_ms"] = (time.perf_counter() - start) * 1000

    if image is None:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        return False, timings

    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    start = time.perf_counter()
    _, faces = detector.detect(image)
    timings["detection_ms"] = (time.perf_counter() - start) * 1000

    if faces is None or len(faces) == 0:
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        return False, timings

    face = max(faces, key=lambda item: item[2] * item[3])

    start = time.perf_counter()
    aligned = recognizer.alignCrop(image, face)
    timings["alignment_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    feature = recognizer.feature(aligned)
    timings["feature_ms"] = (time.perf_counter() - start) * 1000

    timings["total_ms"] = (time.perf_counter() - total_start) * 1000
    return feature is not None, timings


def main():
    parser = argparse.ArgumentParser(
        description="Offline YuNet and SFace performance benchmark."
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.limit < 1 or args.warmup < 0:
        raise SystemExit("Limit must be positive and warmup cannot be negative.")

    for required in (DATASET_DIR, YUNET_PATH, SFACE_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    all_images = sorted(
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not all_images:
        raise RuntimeError("No dataset images found.")

    rng = random.Random(args.seed)
    selected = rng.sample(all_images, min(args.limit, len(all_images)))

    print("Initializing YuNet and SFace models...")
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.9, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

    print(f"Dataset images available: {len(all_images)}")
    print(f"Images selected: {len(selected)}")
    print(f"Warm-up images: {min(args.warmup, len(selected))}")

    for path in selected[:args.warmup]:
        process_image(path, detector, recognizer)

    records = []
    successful = 0
    unreadable = 0

    print("Running timed benchmark...")
    benchmark_start = time.perf_counter()

    for index, path in enumerate(selected, start=1):
        has_face, timings = process_image(path, detector, recognizer)

        if has_face:
            successful += 1
        elif timings["image_read_ms"] > 0 and timings["detection_ms"] > 0:
            # No face was detected; this is a valid negative image.
            pass
        else:
            unreadable += 1

        records.append({
            "image": str(path.relative_to(BASE_DIR)),
            "face_feature_extracted": has_face,
            **{key: round(value, 4) for key, value in timings.items()},
        })

        if index % 50 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} images")

    elapsed = time.perf_counter() - benchmark_start
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

        readable = len(selected) - unreadable
    readable_records = [
        row for row in records
        if row["image_read_ms"] > 0
    ]
    valid = [row for row in records if row["face_feature_extracted"]]

    def average(key, rows):
        return round(
            sum(row[key] for row in rows) / len(rows), 4
        ) if rows else 0.0

    summary = {
        "benchmark": "Offline YuNet + SFace performance",
        "dataset": "Labeled Faces in the Wild (LFW)",
        "protocol": "Deterministic random image sample; not a recognition accuracy test",
        "seed": args.seed,
        "requested_images": len(selected),
        "readable_images": readable,
        "unreadable_images": unreadable,
        "images_with_extracted_features": successful,
        "images_without_extracted_features": readable - successful,
        "warmup_images": min(args.warmup, len(selected)),
        "total_benchmark_seconds": round(elapsed, 4),
        "overall_images_per_second": round(len(selected) / elapsed, 4) if elapsed else 0,
        "average_image_read_ms": average("image_read_ms", records),
        "average_detection_ms": average("detection_ms", readable_records),
        "average_alignment_ms": average("alignment_ms", valid),
        "average_feature_extraction_ms": average("feature_ms", valid),
        "average_end_to_end_ms_all_images": average("total_ms", records),
        "average_end_to_end_ms_with_features": average("total_ms", valid),
        "feature_pipeline_images_per_second": (
            round(1000 / average("total_ms", valid), 4) if valid else 0
        ),
        "note": "Stage averages use readable images for detection and successfully processed faces for alignment and feature extraction. Overall throughput includes all selected images.",
    }

    json_path = OUTPUT_DIR / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n========== PERFORMANCE BENCHMARK ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nCSV saved: {csv_path}")
    print(f"Summary saved: {json_path}")


if __name__ == "__main__":
    main()
