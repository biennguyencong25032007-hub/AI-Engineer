class Student:
    def __init__(self, student_id, name, age, scores=None):
        self.student_id = student_id
        self.name = name
        self.age = age
        self.scores = scores if scores else []

    def add_score(self, score):
        if 0 <= score <= 10:
            self.scores.append(score)
        else:
            print("❌ Điểm không hợp lệ!")

    def remove_score(self, score):
        if score in self.scores:
            self.scores.remove(score)
        else:
            print("❌ Không tìm thấy điểm!")

    def average_score(self):
        return sum(self.scores) / len(self.scores) if self.scores else 0

    def rank(self):
        avg = self.average_score()
        if avg >= 8:
            return "Giỏi"
        elif avg >= 6.5:
            return "Khá"
        elif avg >= 5:
            return "Trung bình"
        else:
            return "Yếu"

    def update_name(self, new_name):
        self.name = new_name

    def display(self):
        print(f"{self.student_id} | {self.name} | {self.average_score():.2f} | {self.rank()}")

# QUẢN LÝ DANH SÁCH
students = []


def find_student(student_id):
    for s in students:
        if s.student_id == student_id:
            return s
    return None


def add_student():
    student_id = input("ID: ")
    name = input("Tên: ")
    age = int(input("Tuổi: "))

    s = Student(student_id, name, age)
    students.append(s)
    print("✅ Đã thêm!")


def add_score():
    student_id = input("Nhập ID: ")
    s = find_student(student_id)

    if s:
        score = float(input("Nhập điểm: "))
        s.add_score(score)
    else:
        print("❌ Không tìm thấy sinh viên")


def show_all():
    print("\n=== DANH SÁCH ===")
    for s in students:
        s.display()


def delete_student():
    student_id = input("ID cần xoá: ")
    s = find_student(student_id)

    if s:
        students.remove(s)
        print("✅ Đã xoá")
    else:
        print("❌ Không tìm thấy")


def update_student():
    student_id = input("ID: ")
    s = find_student(student_id)

    if s:
        new_name = input("Tên mới: ")
        s.update_name(new_name)
        print("✅ Đã cập nhật")
    else:
        print("❌ Không tìm thấy")


# MENU
def menu():
    while True:
        print("""
======== MENU ========
1. Thêm sinh viên
2. Thêm điểm
3. Xem danh sách
4. Xoá sinh viên
5. Sửa tên
6. Thoát
""")

        choice = input("Chọn: ")

        if choice == "1":
            add_student()
        elif choice == "2":
            add_score()
        elif choice == "3":
            show_all()
        elif choice == "4":
            delete_student()
        elif choice == "5":
            update_student()
        elif choice == "6":
            print("👋 Bye!")
            break
        else:
            print("❌ Sai lựa chọn!")

# CHẠY CHƯƠNG TRÌNH

if __name__ == "__main__":
    menu()