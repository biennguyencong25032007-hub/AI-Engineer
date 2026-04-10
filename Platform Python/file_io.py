"""
file_io.py - Xử lý File I/O trong Python
Đọc/ghi JSON, CSV, TXT cho Student Manager
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data.json"
LOG_FILE = Path(__file__).parent.parent.parent / "error.log"


# ========================
# 1. JSON Operations
# ========================

def load_data() -> dict:
    """Đọc toàn bộ dữ liệu từ data.json"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ Đã load dữ liệu từ: {DATA_FILE}")
            return data
    except FileNotFoundError:
        print(f"⚠️  File không tồn tại, tạo mới: {DATA_FILE}")
        default = {"students": [], "courses": []}
        save_data(default)
        return default
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi JSON: {e}")
        return {"students": [], "courses": []}


def save_data(data: dict) -> bool:
    """Lưu dữ liệu vào data.json"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu dữ liệu vào: {DATA_FILE}")
        return True
    except Exception as e:
        log_error(f"Lỗi lưu file: {e}")
        return False


# ========================
# 2. CSV Operations
# ========================

def export_students_csv(students: list, filepath: str = "students_export.csv") -> bool:
    """Xuất danh sách sinh viên ra CSV"""
    if not students:
        print("⚠️  Không có dữ liệu để xuất")
        return False

    fieldnames = ["id", "name", "age", "email", "major", "gpa"]
    try:
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(students)
        print(f"✅ Xuất CSV thành công: {filepath} ({len(students)} sinh viên)")
        return True
    except Exception as e:
        log_error(f"Lỗi xuất CSV: {e}")
        return False


def import_students_csv(filepath: str) -> list:
    """Nhập sinh viên từ file CSV"""
    students = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['age'] = int(row.get('age', 0))
                row['gpa'] = float(row.get('gpa', 0.0))
                students.append(dict(row))
        print(f"✅ Nhập CSV thành công: {len(students)} sinh viên")
        return students
    except FileNotFoundError:
        print(f"❌ File không tồn tại: {filepath}")
        return []
    except Exception as e:
        log_error(f"Lỗi nhập CSV: {e}")
        return []


# ========================
# 3. Log Operations
# ========================

def log_error(message: str):
    """Ghi log lỗi vào error.log"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR: {message}\n"
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"❌ Không thể ghi log: {e}")


def log_info(message: str):
    """Ghi log thông tin"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] INFO:  {message}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def read_log(n_lines: int = 20) -> list:
    """Đọc n dòng cuối cùng của log"""
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-n_lines:]
    except FileNotFoundError:
        return []


# ========================
# 4. File Utilities
# ========================

def backup_data(backup_dir: str = "backups"):
    """Backup data.json với timestamp"""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"data_backup_{timestamp}.json")
    try:
        data = load_data()
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Backup thành công: {backup_file}")
        return backup_file
    except Exception as e:
        log_error(f"Lỗi backup: {e}")
        return None


def get_file_info(filepath: str) -> dict:
    """Lấy thông tin file"""
    path = Path(filepath)
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": path.suffix
    }


# ========================
# 5. Main Demo
# ========================

if __name__ == "__main__":
    print("=" * 50)
    print("       DEMO FILE I/O PYTHON")
    print("=" * 50)

    # Load data
    print("\n📌 Load JSON:")
    data = load_data()
    print(f"   Số sinh viên: {len(data['students'])}")
    print(f"   Số môn học: {len(data['courses'])}")

    # Export CSV
    print("\n📌 Export CSV:")
    export_students_csv(data['students'], "/tmp/students_export.csv")

    # Import CSV
    print("\n📌 Import CSV:")
    imported = import_students_csv("/tmp/students_export.csv")
    print(f"   Imported: {len(imported)} sinh viên")

    # File info
    print("\n📌 Thông tin file:")
    info = get_file_info(str(DATA_FILE))
    for k, v in info.items():
        print(f"   {k}: {v}")

    # Log
    print("\n📌 Ghi log:")
    log_info("Demo file_io.py chạy thành công")
    log_error("Test error logging")
    logs = read_log(5)
    print(f"   {len(logs)} dòng log gần nhất:")
    for line in logs:
        print(f"   {line.strip()}")

    print("\n✅ Demo hoàn tất!")