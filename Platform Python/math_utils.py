def calculate_average(scores):
    return sum(scores) / len(scores) if scores else 0


def get_statistics(scores):
    if not scores:
        return {"avg": 0, "max": None, "min": None}

    return {
        "avg": calculate_average(scores),
        "max": max(scores),
        "min": min(scores)
    }