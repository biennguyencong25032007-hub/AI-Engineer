"""
exception_demo.py - Xử lý ngoại lệ trong Python
Minh họa các kỹ thuật try/except, custom exceptions
"""

# ========================
# 1. Custom Exceptions
# ========================

class StudentError(Exception):
    """Base exception cho Student Manager"""
    pass

class StudentNotFoundError(StudentError):
    def __init__(self, student_id):
        self.student_id = student_id
        super().__init__(f"Không tìm thấy sinh viên với ID: {student_id}")

class InvalidGPAError(StudentError):
    def __init__(self, gpa):
        self.gpa = gpa
        super().__init__(f"GPA không hợp lệ: {gpa}. GPA phải từ 0.0 đến 4.0")

class DuplicateStudentError(StudentError):
    def __init__(self, student_id):
        super().__init__(f"Sinh viên ID '{student_id}' đã tồn tại!")


# ========================
# 2. Demo try/except
# ========================

def chia_so(a, b):
    """Demo ZeroDivisionError"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("❌ Lỗi: Không thể chia cho 0!")
        return None
    except TypeError as e:
        print(f"❌ Lỗi kiểu dữ liệu: {e}")
        return None
    finally:
        print("✅ Khối finally luôn được thực thi")


def doc_file_an_toan(filepath):
    """Demo FileNotFoundError"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ File không tồn tại: {filepath}")
        return ""
    except PermissionError:
        print(f"❌ Không có quyền đọc file: {filepath}")
        return ""
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return ""


def validate_gpa(gpa):
    """Validate GPA với custom exception"""
    try:
        gpa = float(gpa)
        if not 0.0 <= gpa <= 4.0:
            raise InvalidGPAError(gpa)
        return gpa
    except ValueError:
        raise InvalidGPAError(gpa)


def tim_sinh_vien(students: dict, student_id: str):
    """Tìm sinh viên, raise exception nếu không thấy"""
    if student_id not in students:
        raise StudentNotFoundError(student_id)
    return students[student_id]


# ========================
# 3. Context Manager
# ========================

class DatabaseConnection:
    """Minh họa context manager (with statement)"""
    def __init__(self, db_name):
        self.db_name = db_name
        self.connected = False

    def __enter__(self):
        print(f"🔌 Kết nối tới database: {self.db_name}")
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"🔒 Đóng kết nối database: {self.db_name}")
        self.connected = False
        if exc_type is not None:
            print(f"⚠️  Có lỗi xảy ra: {exc_val}")
        return False  # Không suppress exception

    def query(self, sql):
        if not self.connected:
            raise ConnectionError("Chưa kết nối database!")
        print(f"📊 Thực thi query: {sql}")
        return [{"id": 1, "name": "Sample"}]


# ========================
# 4. Main Demo
# ========================

if __name__ == "__main__":
    print("=" * 50)
    print("    DEMO XỬ LÝ NGOẠI LỆ PYTHON")
    print("=" * 50)

    # Demo chia số
    print("\n📌 Demo ZeroDivisionError:")
    print(chia_so(10, 2))
    chia_so(10, 0)

    # Demo file
    print("\n📌 Demo FileNotFoundError:")
    doc_file_an_toan("data.json")
    doc_file_an_toan("file_khong_ton_tai.txt")

    # Demo custom exception
    print("\n📌 Demo Custom Exception:")
    students = {
        "SV001": {"name": "Nguyễn Văn An", "gpa": 3.8},
        "SV002": {"name": "Trần Thị Bình", "gpa": 3.5}
    }
    try:
        sv = tim_sinh_vien(students, "SV001")
        print(f"✅ Tìm thấy: {sv['name']}")
        sv = tim_sinh_vien(students, "SV999")
    except StudentNotFoundError as e:
        print(f"❌ {e}")

    # Demo validate GPA
    print("\n📌 Demo Validate GPA:")
    for gpa in [3.5, 5.0, "abc", -1]:
        try:
            valid = validate_gpa(gpa)
            print(f"✅ GPA hợp lệ: {valid}")
        except InvalidGPAError as e:
            print(f"❌ {e}")

    # Demo context manager
    print("\n📌 Demo Context Manager:")
    with DatabaseConnection("students.db") as db:
        results = db.query("SELECT * FROM students")
        print(f"📋 Kết quả: {results}")

    print("\n✅ Demo hoàn tất!")