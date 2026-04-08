import json
import os
from datetime import datetime

DEFAULT_FILE = "students.json"
BACKUP_FILE = "students_backup.json"


# 🔄 CONVERT OBJECT <-> DICT
def student_to_dict(student):
    return {
        "student_id": student.student_id,
        "name": student.name,
        "age": student.age,
        "scores": student.scores
    }


def dict_to_student(data, StudentClass):
    return StudentClass(
        data["student_id"],
        data["name"],
        data["age"],
        data.get("scores", [])
    )

# 💾 SAVE
def save_students(students, filename=DEFAULT_FILE):
    data = [student_to_dict(s) for s in students]

    try:
        # 🔥 Backup trước khi ghi
        if os.path.exists(filename):
            os.replace(filename, BACKUP_FILE)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("💾 Đã lưu dữ liệu!")

    except Exception as e:
        print("❌ Lỗi khi lưu:", e)

# 📤 LOAD
def load_students(StudentClass, filename=DEFAULT_FILE):
    students = []

    try:
        if not os.path.exists(filename):
            return students

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        students = [dict_to_student(item, StudentClass) for item in data]

    except Exception as e:
        print("⚠️ Lỗi file chính, thử backup...", e)

        # 🔥 Load backup nếu file lỗi
        try:
            with open(BACKUP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            students = [dict_to_student(item, StudentClass) for item in data]
            print("✅ Đã khôi phục từ backup!")
        except:
            print("❌ Backup cũng lỗi luôn!")

    return students

# ⚡ AUTO SAVE
def auto_save(students):
    """Tự động lưu (gọi sau mỗi thay đổi)"""
    save_students(students)

# 🧹 CLEAR FILE
def clear_data(filename=DEFAULT_FILE):
    try:
        with open(filename, "w") as f:
            f.write("[]")
        print("🗑️ Đã xoá dữ liệu!")
    except Exception as e:
        print("❌ Lỗi xoá:", e)