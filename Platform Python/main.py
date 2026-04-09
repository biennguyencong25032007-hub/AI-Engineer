from student_manager import StudentManager
from oop_student import Student
from file_io import save_to_file, load_from_file
from exception_demo import safe_input_int, safe_input_float
from math_utils import get_statistics

DATA_FILE = "students.json"


def load_data(manager):
    data = load_from_file(DATA_FILE)
    manager.students = [Student.from_dict(d) for d in data]


def save_data(manager):
    data = [s.to_dict() for s in manager.students]
    save_to_file(DATA_FILE, data)


def menu():
    print("\n===== STUDENT MANAGER =====")
    print("1. Add student")
    print("2. Show all")
    print("3. Update student")
    print("4. Delete student")
    print("5. Statistics")
    print("6. Save & Exit")


def main():
    manager = StudentManager()
    load_data(manager)

    while True:
        menu()
        choice = input("Choose: ")

        if choice == "1":
            sid = input("ID: ")
            name = input("Name: ")
            age = safe_input_int("Age: ")
            score = safe_input_float("Score: ")
            manager.add(Student(sid, name, age, score))

        elif choice == "2":
            manager.display()

        elif choice == "3":
            sid = input("ID to update: ")
            score = safe_input_float("New score: ")
            if manager.update(sid, score=score):
                print("✅ Updated")
            else:
                print("❌ Not found")

        elif choice == "4":
            sid = input("ID to delete: ")
            manager.remove(sid)

        elif choice == "5":
            stats = get_statistics(manager.get_scores())
            print(stats)

        elif choice == "6":
            save_data(manager)
            print("💾 Saved!")
            break

        else:
            print("❌ Invalid!")


if __name__ == "__main__":
    main()