import logging


logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class InvalidAgeError(Exception):
    pass


class InvalidScoreError(Exception):
    pass


def safe_input_number():
    try:
        x = int(input("Nhập số: "))
        print("Bạn nhập:", x)
    except ValueError:
        print("❌ Phải nhập số!")
    except Exception as e:
        logging.error(e)
        print("❌ Lỗi không xác định")



def divide():
    try:
        a = int(input("a = "))
        b = int(input("b = "))
        print("Kết quả:", a / b)
    except ZeroDivisionError:
        print("❌ Không thể chia cho 0")
    except ValueError:
        print("❌ Nhập sai kiểu dữ liệu")
    finally:
        print("👉 Luôn chạy dòng này")


def check_student(age, score):
    if age < 0 or age > 100:
        raise InvalidAgeError("Tuổi không hợp lệ!")

    if score < 0 or score > 10:
        raise InvalidScoreError("Điểm không hợp lệ!")

    return True


def test_validation():
    try:
        age = int(input("Tuổi: "))
        score = float(input("Điểm: "))

        check_student(age, score)
        print("✅ Dữ liệu hợp lệ")

    except InvalidAgeError as e:
        print("❌", e)
    except InvalidScoreError as e:
        print("❌", e)
    except Exception as e:
        logging.error(e)
        print("❌ Lỗi khác")


def read_file():
    try:
        with open("not_exist.txt", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("❌ File không tồn tại")
    except Exception as e:
        logging.error(e)
        print("❌ Lỗi đọc file")



def login():
    correct_password = "123"
    attempts = 3

    while attempts > 0:
        try:
            pwd = input("Nhập mật khẩu: ")

            if pwd != correct_password:
                raise ValueError("Sai mật khẩu")

            print("✅ Đăng nhập thành công")
            return

        except ValueError as e:
            attempts -= 1
            print(f"❌ {e} | Còn {attempts} lần")

    print("🚫 Hết số lần thử")



def menu():
    while True:
        print("""
======== EXCEPTION DEMO ========
1. Nhập số an toàn
2. Chia số
3. Validate dữ liệu
4. Đọc file
5. Login retry
6. Thoát
""")

        choice = input("Chọn: ")

        if choice == "1":
            safe_input_number()
        elif choice == "2":
            divide()
        elif choice == "3":
            test_validation()
        elif choice == "4":
            read_file()
        elif choice == "5":
            login()
        elif choice == "6":
            print("👋 Bye!")
            break
        else:
            print("❌ Sai lựa chọn!")


if __name__ == "__main__":
    menu()