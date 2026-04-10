"""
main.py - Platform Python Entry Point
Chạy demo tất cả các module trong Platform Python cùng lúc
"""

import sys
import os

# Thêm đường dẫn student_manager vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "student_manager"))


def run_divider(title: str):
    print(f"\n{'━'*55}")
    print(f"   🚀 MODULE: {title}")
    print(f"{'━'*55}")


def run_math_utils():
    run_divider("MATH UTILS")
    from math_utils import (
        calculate_weighted_gpa, classify_gpa,
        class_statistics, gpa_distribution,
        sigmoid, relu, softmax,
        euclidean_distance, cosine_similarity, normalize
    )

    scores  = [8.5, 9.0, 7.5, 8.0]
    credits = [3, 4, 3, 4]
    gpa = calculate_weighted_gpa(scores, credits)
    print(f"  📐 GPA có trọng số: {gpa} → {classify_gpa(gpa)}")

    gpas = [3.8, 3.5, 3.9, 2.8, 3.2, 2.5, 3.6, 3.0, 2.2, 3.7]
    stats = class_statistics(gpas)
    print(f"  📊 Thống kê lớp: mean={stats['mean']}, std={stats['std_dev']}, max={stats['max']}")

    dist = gpa_distribution(gpas)
    print("  📈 Phân phối:")
    for k, v in dist.items():
        print(f"     {'█'*v:<10} {k} ({v} SV)")

    print(f"\n  🤖 AI/ML Math:")
    print(f"     Sigmoid(1.5)     = {sigmoid(1.5):.4f}")
    print(f"     ReLU(-2)         = {relu(-2)}")
    print(f"     Softmax([1,2,3]) = {softmax([1,2,3])}")
    v1, v2 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    print(f"     Cosine Sim       = {cosine_similarity(v1, v2)}")
    print(f"     Normalize([0..100]) = {normalize([0, 25, 50, 75, 100])}")


def run_exception_demo():
    run_divider("EXCEPTION DEMO")
    from exception_demo import (
        chia_so, validate_gpa,
        StudentNotFoundError, InvalidGPAError, DatabaseConnection
    )

    print("  ⚡ ZeroDivisionError:")
    print(f"     10 / 2 = {chia_so(10, 2)}")
    chia_so(5, 0)

    print("\n  ⚡ Custom Exception - Validate GPA:")
    for gpa in [3.8, 4.5, "xyz"]:
        try:
            print(f"     GPA {gpa!r:>5} → {validate_gpa(gpa):.1f} ✅")
        except InvalidGPAError as e:
            print(f"     GPA {gpa!r:>5} → ❌ {e}")

    print("\n  ⚡ Context Manager - DatabaseConnection:")
    with DatabaseConnection("platform_python.db") as db:
        db.query("SELECT * FROM students LIMIT 3")

    print("\n  ⚡ StudentNotFoundError:")
    students = {"SV001": {"name": "An"}, "SV002": {"name": "Bình"}}
    for sid in ["SV001", "SV999"]:
        try:
            if sid not in students:
                raise StudentNotFoundError(sid)
            print(f"     Tìm thấy: {students[sid]['name']} ✅")
        except StudentNotFoundError as e:
            print(f"     ❌ {e}")


def run_file_io():
    run_divider("FILE I/O")
    from file_io import load_data, export_students_csv, log_info, log_error, get_file_info
    from pathlib import Path

    print("  📂 Load JSON:")
    data = load_data()
    print(f"     Sinh viên: {len(data['students'])}, Môn học: {len(data['courses'])}")

    print("\n  📤 Export CSV:")
    export_students_csv(data['students'], "/tmp/platform_demo_export.csv")

    print("\n  📋 File Info:")
    info = get_file_info(str(Path(__file__).parent / "student_manager" / "data.json"))
    print(f"     Tên: {info['name']}, Kích thước: {info['size_kb']} KB")
    print(f"     Sửa lần cuối: {info['modified']}")

    print("\n  📝 Ghi Log:")
    log_info("Platform Python main.py chạy demo thành công")
    log_error("Demo log error từ main.py")
    print("     ✅ Đã ghi vào error.log")


def run_oop_student():
    run_divider("OOP STUDENT")
    from oop_student import Student, GraduateStudent, StudentManager

    manager = StudentManager()

    # Thêm sinh viên
    sv_list = [
        Student("SV001", "Nguyễn Văn An",   20, "an@ai.edu.vn",    "AI Engineer"),
        Student("SV002", "Trần Thị Bình",   21, "binh@ai.edu.vn",  "Data Science"),
        Student("SV003", "Lê Minh Cường",   22, "cuong@ai.edu.vn", "ML Engineer"),
        Student("SV004", "Phạm Thị Dung",   20, "dung@ai.edu.vn",  "AI Engineer"),
        GraduateStudent("MS001", "Hoàng Văn Em", 25,
                        "em@ai.edu.vn", "AI Research",
                        "LLM trong Giáo dục Việt Nam"),
    ]

    print("  👥 Thêm sinh viên:")
    for sv in sv_list:
        manager.add_student(sv)

    # Thêm điểm
    data_scores = {
        "SV001": [("Python", 8.5, 3), ("ML", 9.0, 4), ("DeepLearn", 8.0, 4), ("NLP", 7.5, 3)],
        "SV002": [("Python", 7.0, 3), ("ML", 6.5, 4), ("SQL", 8.0, 3)],
        "SV003": [("Python", 9.5, 3), ("ML", 9.0, 4), ("CV", 8.5, 3)],
        "SV004": [("Python", 7.5, 3), ("ML", 7.0, 4), ("MLOps", 8.0, 3)],
        "MS001": [("Advanced ML", 9.0, 4), ("Research Method", 8.5, 3)],
    }
    print("\n  📝 Nhập điểm (tự động)...")
    for sid, scores in data_scores.items():
        sv = manager.get_student(sid)
        if sv:
            for course, score, credits in scores:
                sv.add_score(course, score, credits)

    # Bảng điểm
    print(f"\n  📜 Bảng điểm SV001:")
    sv = manager.get_student("SV001")
    print(sv.get_transcript())

    # Tổng kết
    manager.summary()

    print(f"  📊 Tổng sinh viên đã khởi tạo: {Student.get_count()}")


# ========================
# MAIN
# ========================

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║       PLATFORM PYTHON - AI ENGINEER COURSE           ║
║       Demo tất cả modules cùng lúc                   ║
╚══════════════════════════════════════════════════════╝
    """)

    modules = [
        ("Math Utils",       run_math_utils),
        ("Exception Demo",   run_exception_demo),
        ("File I/O",         run_file_io),
        ("OOP Student",      run_oop_student),
    ]

    for name, func in modules:
        try:
            func()
        except Exception as e:
            print(f"\n  ❌ Lỗi trong module {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'━'*55}")
    print("  ✅ Hoàn thành demo toàn bộ Platform Python!")
    print(f"{'━'*55}\n")


if __name__ == "__main__":
    main()