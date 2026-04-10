from datetime import datetime
from typing import List, Optional
from math_utils import classify_gpa, calculate_weighted_gpa


# ========================
# 1. Class Course
# ========================

class Course:
    """Đại diện cho một môn học"""

    def __init__(self, code: str, name: str, credits: int):
        self.code = code
        self.name = name
        self.credits = credits

    def __repr__(self):
        return f"Course(code='{self.code}', name='{self.name}', credits={self.credits})"

    def __str__(self):
        return f"[{self.code}] {self.name} ({self.credits} tín chỉ)"

    def to_dict(self) -> dict:
        return {"code": self.code, "name": self.name, "credits": self.credits}


# ========================
# 2. Class Student
# ========================

class Student:
    """Đại diện cho một sinh viên"""

    _count = 0  # Class variable đếm tổng sinh viên

    def __init__(self, student_id: str, name: str, age: int,
                 email: str, major: str):
        self.id = student_id
        self.name = name
        self.age = age
        self.email = email
        self.major = major
        self._scores: dict = {}      # {course_code: score}
        self._credits: dict = {}     # {course_code: credits}
        self.created_at = datetime.now()
        Student._count += 1

    # --- Properties ---
    @property
    def gpa(self) -> float:
        """Tính GPA tự động từ điểm số"""
        if not self._scores:
            return 0.0
        scores = list(self._scores.values())
        credits = [self._credits.get(c, 3) for c in self._scores.keys()]
        return calculate_weighted_gpa(scores, credits)

    @property
    def rank(self) -> str:
        """Xếp loại học lực"""
        return classify_gpa(self.gpa)

    @property
    def total_credits(self) -> int:
        return sum(self._credits.values())

    # --- Methods ---
    def add_score(self, course_code: str, score: float, credits: int = 3):
        """Thêm điểm môn học"""
        if not 0 <= score <= 10:
            raise ValueError(f"Điểm phải từ 0-10, nhận được: {score}")
        self._scores[course_code] = score
        self._credits[course_code] = credits
        print(f"✅ {self.name}: {course_code} = {score}/10 ({credits} TC)")

    def get_scores(self) -> dict:
        return dict(self._scores)

    def get_transcript(self) -> str:
        """Bảng điểm sinh viên"""
        lines = [
            f"{'='*45}",
            f"  BẢNG ĐIỂM - {self.name}",
            f"  Mã SV: {self.id} | Ngành: {self.major}",
            f"{'='*45}",
            f"  {'Môn học':<20} {'Điểm':>6} {'TC':>4}",
            f"  {'-'*35}",
        ]
        for course, score in self._scores.items():
            tc = self._credits.get(course, 3)
            lines.append(f"  {course:<20} {score:>6.1f} {tc:>4}")
        lines += [
            f"  {'-'*35}",
            f"  Tổng tín chỉ: {self.total_credits}",
            f"  GPA: {self.gpa:.2f}/4.0  →  {self.rank}",
            f"{'='*45}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "email": self.email,
            "major": self.major,
            "gpa": self.gpa,
            "courses": list(self._scores.keys())
        }

    def __str__(self):
        return f"Student({self.id}: {self.name}, GPA={self.gpa:.2f})"

    def __repr__(self):
        return f"Student(id='{self.id}', name='{self.name}')"

    def __lt__(self, other):
        return self.gpa < other.gpa

    # Class method
    @classmethod
    def get_count(cls) -> int:
        return cls._count

    @classmethod
    def from_dict(cls, data: dict) -> 'Student':
        """Tạo Student từ dictionary"""
        return cls(
            student_id=data['id'],
            name=data['name'],
            age=data.get('age', 0),
            email=data.get('email', ''),
            major=data.get('major', '')
        )

    # Static method
    @staticmethod
    def validate_email(email: str) -> bool:
        return '@' in email and '.' in email.split('@')[-1]


# ========================
# 3. Class Enrollment (Kế thừa ví dụ)
# ========================

class GraduateStudent(Student):
    """Sinh viên sau đại học - Kế thừa từ Student"""

    def __init__(self, student_id, name, age, email, major, thesis_topic: str):
        super().__init__(student_id, name, age, email, major)
        self.thesis_topic = thesis_topic
        self.thesis_score: Optional[float] = None

    def submit_thesis(self, score: float):
        if not 0 <= score <= 10:
            raise ValueError("Điểm luận văn phải từ 0-10")
        self.thesis_score = score
        print(f"📜 {self.name} nộp luận văn: {score}/10")

    def __str__(self):
        base = super().__str__()
        return f"Graduate{base} | Thesis: {self.thesis_topic}"


# ========================
# 4. Class StudentManager
# ========================

class StudentManager:
    """Quản lý danh sách sinh viên"""

    def __init__(self):
        self._students: dict[str, Student] = {}
        self._courses: dict[str, Course] = {}

    # CRUD Operations
    def add_student(self, student: Student) -> bool:
        if student.id in self._students:
            print(f"❌ Sinh viên {student.id} đã tồn tại!")
            return False
        if not Student.validate_email(student.email):
            print(f"❌ Email không hợp lệ: {student.email}")
            return False
        self._students[student.id] = student
        print(f"✅ Thêm sinh viên: {student.name}")
        return True

    def get_student(self, student_id: str) -> Optional[Student]:
        return self._students.get(student_id)

    def update_student(self, student_id: str, **kwargs) -> bool:
        student = self.get_student(student_id)
        if not student:
            print(f"❌ Không tìm thấy: {student_id}")
            return False
        for key, value in kwargs.items():
            if hasattr(student, key):
                setattr(student, key, value)
        print(f"✅ Cập nhật {student_id}: {kwargs}")
        return True

    def delete_student(self, student_id: str) -> bool:
        if student_id not in self._students:
            print(f"❌ Không tìm thấy: {student_id}")
            return False
        name = self._students[student_id].name
        del self._students[student_id]
        print(f"🗑️  Đã xóa: {name}")
        return True

    def search(self, keyword: str) -> List[Student]:
        """Tìm kiếm sinh viên theo tên hoặc ID"""
        kw = keyword.lower()
        return [
            s for s in self._students.values()
            if kw in s.name.lower() or kw in s.id.lower()
        ]

    def get_top_students(self, n: int = 3) -> List[Student]:
        """Lấy top N sinh viên GPA cao nhất"""
        return sorted(self._students.values(), reverse=True)[:n]

    def get_all_students(self) -> List[Student]:
        return list(self._students.values())

    def summary(self):
        """In báo cáo tổng kết"""
        students = self.get_all_students()
        if not students:
            print("Không có sinh viên nào!")
            return
        gpas = [s.gpa for s in students]
        print(f"\n{'='*45}")
        print(f"  TỔNG KẾT LỚP AI ENGINEER")
        print(f"{'='*45}")
        print(f"  Tổng sinh viên: {len(students)}")
        print(f"  GPA trung bình: {sum(gpas)/len(gpas):.2f}")
        print(f"  GPA cao nhất:  {max(gpas):.2f}")
        print(f"  GPA thấp nhất: {min(gpas):.2f}")
        print(f"\n  🏆 Top 3 sinh viên:")
        for i, s in enumerate(self.get_top_students(3), 1):
            print(f"     {i}. {s.name} - GPA: {s.gpa:.2f} ({s.rank})")
        print(f"{'='*45}\n")


# ========================
# 5. Main Demo
# ========================

if __name__ == "__main__":
    print("=" * 50)
    print("    DEMO OOP - STUDENT MANAGER")
    print("=" * 50)

    # Tạo manager
    manager = StudentManager()

    # Tạo sinh viên
    students_data = [
        ("SV001", "Nguyễn Văn An", 20, "an@email.com", "AI Engineer"),
        ("SV002", "Trần Thị Bình", 21, "binh@email.com", "Data Science"),
        ("SV003", "Lê Minh Cường", 22, "cuong@email.com", "ML Engineer"),
        ("SV004", "Phạm Thị Dung", 20, "dung@email.com", "AI Engineer"),
    ]

    print("\n📌 Thêm sinh viên:")
    for data in students_data:
        s = Student(*data)
        manager.add_student(s)

    # Thêm điểm
    print("\n📌 Nhập điểm:")
    sv = manager.get_student("SV001")
    sv.add_score("Python",    8.5, 3)
    sv.add_score("ML",        9.0, 4)
    sv.add_score("DeepLearn", 8.0, 4)
    sv.add_score("NLP",       7.5, 3)

    sv2 = manager.get_student("SV002")
    sv2.add_score("Python", 7.0, 3)
    sv2.add_score("ML",     6.5, 4)
    sv2.add_score("SQL",    8.0, 3)

    sv3 = manager.get_student("SV003")
    sv3.add_score("Python", 9.5, 3)
    sv3.add_score("ML",     9.0, 4)
    sv3.add_score("CV",     8.5, 3)

    # Bảng điểm
    print("\n📌 Bảng điểm SV001:")
    print(sv.get_transcript())

    # Tìm kiếm
    print("\n📌 Tìm kiếm 'nguyễn':")
    results = manager.search("nguyễn")
    for r in results:
        print(f"   → {r}")

    # Graduate student
    print("\n📌 Sinh viên sau đại học:")
    grad = GraduateStudent("MS001", "Hoàng Văn Em", 25,
                           "em@email.com", "AI Research",
                           "Ứng dụng LLM trong Giáo dục")
    grad.add_score("Advanced ML", 9.0, 4)
    grad.submit_thesis(8.7)
    manager.add_student(grad)
    print(f"   {grad}")

    # Summary
    manager.summary()

    print(f"📊 Tổng sinh viên đã tạo: {Student.get_count()}")
    print("\n✅ Demo OOP hoàn tất!")