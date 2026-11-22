"""
LFW gallery/probe evaluation for the Face Recognition System.

Custom open-set identification experiment; NOT the official LFW 6,000-pair
verification protocol.

Run from the project root:
    .venv\\Scripts\\python.exe evaluation\\evaluate_lfw.py
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "evaluation" / "public_dataset" / "lfw"
MODEL_DIR = BASE_DIR / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "lfw_results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_THRESHOLD = 0.45
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def image_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_feature(image_path, detector, recognizer):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda item: item[2] * item[3])
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return feature if feature is not None else None


def safe_divide(a, b):
    return a / b if b else 0.0


def predict(feature, gallery, recognizer):
    best_name, best_score = "Unknown", -1.0
    for name, reference_feature in gallery:
        score = float(recognizer.match(
            feature, reference_feature, cv2.FaceRecognizerSF_FR_COSINE
        ))
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def calculate_metrics(records, threshold):
    tp = tn = fp = fn = 0
    correct_id = wrong_id = known_rejected = 0
    for record in records:
        predicted = (
            record["best_name"]
            if record["best_score"] >= threshold else "Unknown"
        )
        if record["category"] == "Known":
            if predicted == "Unknown":
                fn += 1
                known_rejected += 1
            else:
                tp += 1
                if predicted == record["expected"]:
                    correct_id += 1
                else:
                    wrong_id += 1
        elif predicted == "Unknown":
            tn += 1
        else:
            fp += 1

    known_total, unknown_total = tp + fn, fp + tn
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "threshold": threshold,
        "TP_known_accepted": tp,
        "TN_unknown_rejected": tn,
        "FP_unknown_accepted": fp,
        "FN_known_rejected": fn,
        "known_correct_identity": correct_id,
        "known_wrong_identity": wrong_id,
        "known_false_rejected": known_rejected,
        "known_probes": known_total,
        "unknown_probes": unknown_total,
        "binary_acceptance_accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1_score": safe_divide(2 * precision * recall, precision + recall),
        "FAR": safe_divide(fp, unknown_total),
        "FRR": safe_divide(fn, known_total),
        "exact_known_identification_rate": safe_divide(correct_id, known_total),
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Run a reproducible custom LFW gallery/probe experiment."
    )
    parser.add_argument("--enrolled", type=int, default=100)
    parser.add_argument("--unknown", type=int, default=100)
    parser.add_argument("--max-probes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    excluded_identities = set()
    if args.exclude_manifest:
        if not args.exclude_manifest.is_file():
            raise FileNotFoundError(f"Exclusion manifest not found: {args.exclude_manifest}")
        with args.exclude_manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "identity" not in (reader.fieldnames or []):
                raise ValueError("Exclusion manifest must contain an identity column.")
            excluded_identities = {row["identity"] for row in reader if row.get("identity")}

    if min(args.enrolled, args.unknown, args.max_probes) < 1:
        raise SystemExit("enrolled, unknown, and max-probes must be positive.")

    for path in (DATASET_DIR, YUNET_PATH, SFACE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    folders = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    identity_images = {p.name: image_files(p) for p in folders}
    eligible = sorted(
        name for name, imgs in identity_images.items()
        if len(imgs) >= 2 and name not in excluded_identities
    )
    if len(eligible) < args.enrolled:
        raise RuntimeError(f"Only {len(eligible)} identities have at least 2 images.")

    rng = random.Random(args.seed)
    enrolled_names = sorted(rng.sample(eligible, args.enrolled))
    enrolled_set = set(enrolled_names)
    unknown_eligible = sorted(
        name for name, imgs in identity_images.items()
        if name not in enrolled_set and name not in excluded_identities and imgs
    )
    if len(unknown_eligible) < args.unknown:
        raise RuntimeError("Not enough identities available for unknown probes.")
    unknown_names = sorted(rng.sample(unknown_eligible, args.unknown))

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.9, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gallery = []
    known_probe_paths = []
    manifest = []
    skipped_gallery = []

    print("Building gallery and selecting known probes...")
    for identity in enrolled_names:
        candidates = identity_images[identity][:]
        rng.shuffle(candidates)

        gallery_path = None
        gallery_feature = None
        for candidate in candidates:
            feature = extract_feature(candidate, detector, recognizer)
            if feature is not None:
                gallery_path, gallery_feature = candidate, feature
                break

        if gallery_feature is None:
            skipped_gallery.append(identity)
            continue

        gallery.append((identity, gallery_feature))
        manifest.append({
            "role": "gallery",
            "identity": identity,
            "image_path": str(gallery_path.relative_to(BASE_DIR)),
        })

        remaining = [p for p in candidates if p != gallery_path]
        chosen_probes = remaining[:args.max_probes]
        for probe in chosen_probes:
            known_probe_paths.append((identity, probe))
            manifest.append({
                "role": "known_probe",
                "identity": identity,
                "image_path": str(probe.relative_to(BASE_DIR)),
            })

    if not gallery:
        raise RuntimeError("No valid gallery faces could be extracted.")

    unknown_probe_paths = []
    print("Selecting unknown probes...")
    for identity in unknown_names:
        candidates = identity_images[identity][:]
        rng.shuffle(candidates)
        unknown_probe_paths.append((identity, candidates[0]))
        manifest.append({
            "role": "unknown_probe",
            "identity": identity,
            "image_path": str(candidates[0].relative_to(BASE_DIR)),
        })

    records = []
    skipped_known = skipped_unknown = 0

    print("Evaluating known probes...")
    for identity, path in known_probe_paths:
        feature = extract_feature(path, detector, recognizer)
        if feature is None:
            skipped_known += 1
            continue
        name, score = predict(feature, gallery, recognizer)
        records.append({
            "category": "Known",
            "image_path": str(path.relative_to(BASE_DIR)),
            "expected": identity,
            "best_name": name,
            "best_score": score,
        })

    print("Evaluating unknown probes...")
    for identity, path in unknown_probe_paths:
        feature = extract_feature(path, detector, recognizer)
        if feature is None:
            skipped_unknown += 1
            continue
        name, score = predict(feature, gallery, recognizer)
        records.append({
            "category": "Unknown",
            "image_path": str(path.relative_to(BASE_DIR)),
            "expected": "Unknown",
            "best_name": name,
            "best_score": score,
        })

    if not records:
        raise RuntimeError("No valid probe images were evaluated.")

    write_csv(
        OUTPUT_DIR / "lfw_split_manifest.csv",
        ["role", "identity", "image_path"],
        manifest,
    )

    detailed = []
    for item in records:
        predicted = item["best_name"] if item["best_score"] >= args.threshold else "Unknown"
        if item["category"] == "Known":
            outcome = (
                "False Rejection" if predicted == "Unknown"
                else "Correct Identification" if predicted == item["expected"]
                else "Wrong Identity"
            )
        else:
            outcome = "Correct Rejection" if predicted == "Unknown" else "False Acceptance"
        detailed.append({
            "Category": item["category"],
            "Image": item["image_path"],
            "Expected": item["expected"],
            "BestMatchedIdentity": item["best_name"],
            "PredictedAtSelectedThreshold": predicted,
            "BestCosineSimilarity": round(item["best_score"], 6),
            "Outcome": outcome,
        })

    write_csv(
        OUTPUT_DIR / "lfw_detailed_results.csv",
        ["Category", "Image", "Expected", "BestMatchedIdentity",
         "PredictedAtSelectedThreshold", "BestCosineSimilarity", "Outcome"],
        detailed,
    )

    sweep = [
        calculate_metrics(records, threshold) for threshold in THRESHOLDS
    ]
    write_csv(
        OUTPUT_DIR / "lfw_threshold_sweep.csv",
        list(sweep[0].keys()),
        sweep,
    )

    metrics = calculate_metrics(records, args.threshold)
    summary = {
        "dataset": "Labeled Faces in the Wild (LFW)",
        "protocol": (
            "Custom open-set gallery/probe identification experiment; "
            "not the official LFW 6000-pair verification protocol."
        ),
        "seed": args.seed,
        "excluded_identity_count": len(excluded_identities),
        "exclusion_manifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "requested_enrolled_identities": args.enrolled,
        "valid_gallery_identities": len(gallery),
        "requested_unknown_identities": args.unknown,
        "max_known_probes_per_identity": args.max_probes,
        "selected_threshold": args.threshold,
        "skipped_gallery_identities": skipped_gallery,
        "skipped_known_probe_images": skipped_known,
        "skipped_unknown_probe_images": skipped_unknown,
        "metrics": metrics,
    }
    with (OUTPUT_DIR / "lfw_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n========== LFW EVALUATION SUMMARY ==========")
    print(f"Dataset identities: {len(folders)}")
    print(f"Dataset images: {sum(len(v) for v in identity_images.values())}")
    print(f"Valid gallery identities: {len(gallery)}")
    print(f"Known probes evaluated: {metrics['known_probes']}")
    print(f"Unknown probes evaluated: {metrics['unknown_probes']}")
    print(f"Skipped gallery identities: {len(skipped_gallery)}")
    print(f"Skipped known probes: {skipped_known}")
    print(f"Skipped unknown probes: {skipped_unknown}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Binary acceptance accuracy: {metrics['binary_acceptance_accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-score: {metrics['f1_score']:.2%}")
    print(f"FAR: {metrics['FAR']:.2%}")
    print(f"FRR: {metrics['FRR']:.2%}")
    print(f"Exact known identification rate: {metrics['exact_known_identification_rate']:.2%}")
    print(f"Known wrong identities: {metrics['known_wrong_identity']}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
