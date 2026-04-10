"""
main.py - Student Manager System
Entry point chính của ứng dụng quản lý sinh viên AI Engineer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from file_io import load_data, save_data, export_students_csv, log_info
from oop_student import Student, StudentManager, Course
from math_utils import class_statistics, gpa_distribution, rank_students
from exception_demo import StudentNotFoundError, InvalidGPAError


# ========================
# MENU HỆ THỐNG
# ========================

def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║        AI ENGINEER - STUDENT MANAGER v1.0        ║
║        Hệ thống Quản lý Sinh viên AI             ║
╚══════════════════════════════════════════════════╝
""")

def print_menu():
    print("""
┌──────────────────────────────────────┐
│           MENU CHÍNH                 │
├──────────────────────────────────────┤
│  1. Xem danh sách sinh viên          │
│  2. Thêm sinh viên mới               │
│  3. Tìm kiếm sinh viên               │
│  4. Xem bảng điểm                    │
│  5. Thống kê lớp học                 │
│  6. Xuất báo cáo CSV                 │
│  7. Lưu dữ liệu                      │
│  0. Thoát                            │
└──────────────────────────────────────┘
""")


# ========================
# KHỞI TẠO DỮ LIỆU
# ========================

def initialize_manager() -> StudentManager:
    """Load dữ liệu từ JSON và khởi tạo StudentManager"""
    manager = StudentManager()
    data = load_data()

    # Khởi tạo sinh viên mẫu nếu file có dữ liệu
    for s_data in data.get("students", []):
        student = Student.from_dict(s_data)
        # Thêm điểm mẫu theo GPA đã lưu (demo)
        gpa = s_data.get("gpa", 3.0)
        # Tính ngược điểm thang 10 tương đương
        score_10 = gpa / 4.0 * 10
        for course in s_data.get("courses", []):
            student.add_score(course, min(score_10 + 0.5, 10.0), 3)
        manager.add_student(student)

    log_info(f"Khởi tạo hệ thống với {len(data.get('students', []))} sinh viên")
    return manager


# ========================
# CÁC CHỨC NĂNG
# ========================

def show_all_students(manager: StudentManager):
    students = manager.get_all_students()
    if not students:
        print("\n⚠️  Chưa có sinh viên nào!\n")
        return

    print(f"\n{'─'*65}")
    print(f"  {'STT':<4} {'Mã SV':<8} {'Họ tên':<22} {'Ngành':<18} {'GPA':>5} {'XL'}")
    print(f"{'─'*65}")
    for i, s in enumerate(rank_students(students), 1):
        print(f"  {i:<4} {s.id:<8} {s.name:<22} {s.major:<18} {s.gpa:>5.2f} {s.rank}")
    print(f"{'─'*65}")
    print(f"  Tổng: {len(students)} sinh viên\n")


def add_new_student(manager: StudentManager):
    print("\n── THÊM SINH VIÊN MỚI ──")
    try:
        sid   = input("  Mã SV: ").strip()
        name  = input("  Họ tên: ").strip()
        age   = int(input("  Tuổi: ").strip())
        email = input("  Email: ").strip()
        major = input("  Ngành học: ").strip()

        student = Student(sid, name, age, email, major)

        # Nhập điểm
        add_scores = input("  Nhập điểm ngay? (y/n): ").strip().lower()
        if add_scores == 'y':
            while True:
                course = input("  Tên môn (Enter để bỏ qua): ").strip()
                if not course:
                    break
                try:
                    score = float(input(f"  Điểm {course} (0-10): "))
                    credits = int(input(f"  Số tín chỉ: "))
                    student.add_score(course, score, credits)
                except (ValueError, InvalidGPAError) as e:
                    print(f"  ❌ {e}")

        manager.add_student(student)

    except ValueError as e:
        print(f"❌ Dữ liệu không hợp lệ: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


def search_student(manager: StudentManager):
    keyword = input("\n  🔍 Nhập từ khóa tìm kiếm: ").strip()
    results = manager.search(keyword)
    if not results:
        print(f"  ⚠️  Không tìm thấy sinh viên nào với từ khóa '{keyword}'")
    else:
        print(f"\n  Tìm thấy {len(results)} sinh viên:")
        for s in results:
            print(f"  → {s.id}: {s.name} | {s.major} | GPA: {s.gpa:.2f}")
    print()


def show_transcript(manager: StudentManager):
    sid = input("\n  Nhập mã sinh viên: ").strip()
    student = manager.get_student(sid)
    if not student:
        print(f"  ❌ Không tìm thấy sinh viên: {sid}\n")
        return
    print(student.get_transcript())


def show_statistics(manager: StudentManager):
    students = manager.get_all_students()
    if not students:
        print("\n⚠️  Chưa có dữ liệu!\n")
        return

    gpas = [s.gpa for s in students if s.gpa > 0]
    if not gpas:
        print("\n⚠️  Chưa có điểm số!\n")
        return

    stats = class_statistics(gpas)
    dist  = gpa_distribution(gpas)

    print(f"\n{'─'*40}")
    print(f"  📊 THỐNG KÊ LỚP AI ENGINEER")
    print(f"{'─'*40}")
    print(f"  Số sinh viên có điểm: {stats['count']}")
    print(f"  GPA trung bình:       {stats['mean']:.3f}")
    print(f"  GPA trung vị:         {stats['median']:.3f}")
    print(f"  Độ lệch chuẩn:        {stats['std_dev']:.3f}")
    print(f"  GPA cao nhất:         {stats['max']:.2f}")
    print(f"  GPA thấp nhất:        {stats['min']:.2f}")

    print(f"\n  📈 PHÂN PHỐI XẾP LOẠI:")
    for category, count in dist.items():
        bar = "█" * count + "░" * (stats['count'] - count)
        pct = count / stats['count'] * 100
        print(f"  {category:<25}: {bar} {count} ({pct:.0f}%)")

    print(f"\n  🏆 TOP 3 SINH VIÊN:")
    for i, s in enumerate(manager.get_top_students(3), 1):
        print(f"  {i}. {s.name} ({s.id}) - GPA: {s.gpa:.2f} - {s.rank}")
    print(f"{'─'*40}\n")


def export_report(manager: StudentManager):
    students = manager.get_all_students()
    filepath = "students_report.csv"
    data = [s.to_dict() for s in students]
    if export_students_csv(data, filepath):
        print(f"  ✅ Đã xuất báo cáo: {filepath}\n")
    else:
        print(f"  ❌ Xuất thất bại!\n")


def save_all_data(manager: StudentManager):
    students = manager.get_all_students()
    data = {
        "students": [s.to_dict() for s in students],
        "courses": []
    }
    if save_data(data):
        print(f"  ✅ Đã lưu {len(students)} sinh viên\n")
    else:
        print("  ❌ Lưu thất bại!\n")


# ========================
# MAIN LOOP
# ========================

def main():
    print_banner()
    print("  🔄 Đang khởi tạo hệ thống...")
    manager = initialize_manager()
    print(f"  ✅ Sẵn sàng! Đã load {len(manager.get_all_students())} sinh viên\n")

    actions = {
        '1': show_all_students,
        '2': add_new_student,
        '3': search_student,
        '4': show_transcript,
        '5': show_statistics,
        '6': export_report,
        '7': save_all_data,
    }

    while True:
        print_menu()
        choice = input("  👉 Chọn chức năng (0-7): ").strip()

        if choice == '0':
            save_all_data(manager)
            print("  👋 Tạm biệt! Hẹn gặp lại.\n")
            break
        elif choice in actions:
            actions[choice](manager)
        else:
            print("  ⚠️  Lựa chọn không hợp lệ!\n")


if __name__ == "__main__":
    main()