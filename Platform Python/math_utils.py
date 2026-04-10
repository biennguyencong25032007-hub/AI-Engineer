"""
math_utils.py - Các hàm toán học hỗ trợ cho AI Engineer
Thống kê, tính toán GPA, xếp loại sinh viên
"""

import math
import statistics
from typing import List, Optional


# ========================
# 1. GPA & Xếp loại
# ========================

def classify_gpa(gpa: float) -> str:
    """Xếp loại học lực dựa trên GPA (thang 4.0)"""
    if gpa >= 3.6:
        return "Xuất sắc 🏆"
    elif gpa >= 3.2:
        return "Giỏi 🥇"
    elif gpa >= 2.5:
        return "Khá 🥈"
    elif gpa >= 2.0:
        return "Trung bình 🥉"
    else:
        return "Yếu ❌"


def calculate_weighted_gpa(scores: List[float], credits: List[int]) -> float:
    """
    Tính GPA có trọng số theo tín chỉ
    Args:
        scores: Điểm từng môn (thang 10)
        credits: Số tín chỉ từng môn
    Returns:
        GPA thang 4.0
    """
    if len(scores) != len(credits):
        raise ValueError("Số môn học và số tín chỉ phải bằng nhau")
    if not scores:
        return 0.0

    total_weighted = sum(s * c for s, c in zip(scores, credits))
    total_credits = sum(credits)

    if total_credits == 0:
        return 0.0

    avg_10 = total_weighted / total_credits
    return round(convert_to_gpa4(avg_10), 2)


def convert_to_gpa4(score_10: float) -> float:
    """Chuyển điểm thang 10 sang thang 4.0"""
    if score_10 >= 9.0:   return 4.0
    elif score_10 >= 8.5: return 3.7
    elif score_10 >= 8.0: return 3.5
    elif score_10 >= 7.5: return 3.2
    elif score_10 >= 7.0: return 3.0
    elif score_10 >= 6.5: return 2.7
    elif score_10 >= 6.0: return 2.5
    elif score_10 >= 5.5: return 2.3
    elif score_10 >= 5.0: return 2.0
    else:                  return 0.0


# ========================
# 2. Thống kê lớp học
# ========================

def class_statistics(gpas: List[float]) -> dict:
    """Tính thống kê mô tả cho lớp học"""
    if not gpas:
        return {}

    return {
        "count": len(gpas),
        "mean": round(statistics.mean(gpas), 3),
        "median": round(statistics.median(gpas), 3),
        "std_dev": round(statistics.stdev(gpas), 3) if len(gpas) > 1 else 0.0,
        "min": min(gpas),
        "max": max(gpas),
        "range": round(max(gpas) - min(gpas), 3),
        "variance": round(statistics.variance(gpas), 3) if len(gpas) > 1 else 0.0,
    }


def gpa_distribution(gpas: List[float]) -> dict:
    """Phân phối xếp loại học lực"""
    distribution = {
        "Xuất sắc (≥3.6)": 0,
        "Giỏi (3.2-3.59)": 0,
        "Khá (2.5-3.19)": 0,
        "Trung bình (2.0-2.49)": 0,
        "Yếu (<2.0)": 0
    }
    for gpa in gpas:
        if gpa >= 3.6:   distribution["Xuất sắc (≥3.6)"] += 1
        elif gpa >= 3.2: distribution["Giỏi (3.2-3.59)"] += 1
        elif gpa >= 2.5: distribution["Khá (2.5-3.19)"] += 1
        elif gpa >= 2.0: distribution["Trung bình (2.0-2.49)"] += 1
        else:            distribution["Yếu (<2.0)"] += 1
    return distribution


def rank_students(students: list, key: str = "gpa", reverse: bool = True) -> list:
    """Xếp hạng sinh viên theo tiêu chí"""
    return sorted(students, key=lambda s: s.get(key, 0), reverse=reverse)


# ========================
# 3. Toán học AI/ML
# ========================

def sigmoid(x: float) -> float:
    """Hàm kích hoạt Sigmoid"""
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """Hàm kích hoạt ReLU"""
    return max(0, x)


def softmax(values: List[float]) -> List[float]:
    """Hàm Softmax cho phân loại"""
    exp_vals = [math.exp(v) for v in values]
    total = sum(exp_vals)
    return [round(e / total, 4) for e in exp_vals]


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Khoảng cách Euclidean giữa 2 vector"""
    if len(v1) != len(v2):
        raise ValueError("Hai vector phải có cùng chiều dài")
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2))), 4)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Độ tương đồng Cosine"""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 4)


def normalize(values: List[float]) -> List[float]:
    """Chuẩn hóa Min-Max về [0, 1]"""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [0.0] * len(values)
    return [round((v - min_v) / (max_v - min_v), 4) for v in values]


# ========================
# 4. Main Demo
# ========================

if __name__ == "__main__":
    print("=" * 50)
    print("      DEMO MATH UTILS - AI ENGINEER")
    print("=" * 50)

    # GPA
    print("\n📌 Tính GPA có trọng số:")
    scores = [8.5, 7.0, 9.0, 6.5, 8.0]
    credits = [3, 4, 4, 3, 3]
    gpa = calculate_weighted_gpa(scores, credits)
    print(f"   Điểm: {scores}")
    print(f"   Tín chỉ: {credits}")
    print(f"   GPA: {gpa} → {classify_gpa(gpa)}")

    # Thống kê lớp
    print("\n📌 Thống kê lớp học:")
    gpas = [3.8, 3.5, 3.9, 2.8, 3.2, 2.5, 3.6, 3.0]
    stats = class_statistics(gpas)
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Phân phối
    print("\n📌 Phân phối xếp loại:")
    dist = gpa_distribution(gpas)
    for k, v in dist.items():
        bar = "█" * v
        print(f"   {k:25s}: {bar} ({v})")

    # AI/ML Math
    print("\n📌 Toán học AI/ML:")
    print(f"   Sigmoid(0)   = {sigmoid(0)}")
    print(f"   Sigmoid(2)   = {sigmoid(2):.4f}")
    print(f"   ReLU(-3)     = {relu(-3)}")
    print(f"   ReLU(5)      = {relu(5)}")
    print(f"   Softmax([1,2,3]) = {softmax([1,2,3])}")

    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    print(f"   Euclidean({v1}, {v2}) = {euclidean_distance(v1, v2)}")
    print(f"   Cosine Sim = {cosine_similarity(v1, v2)}")
    print(f"   Normalize([10,20,30,40]) = {normalize([10,20,30,40])}")

    print("\n✅ Demo hoàn tất!")