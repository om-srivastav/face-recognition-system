THRESHOLD = 0.45

def classify_match(best_name, best_score, threshold=THRESHOLD):
    """Classify a face using a similarity score."""

    if best_score < threshold:
        return "Unknown"

    return best_name