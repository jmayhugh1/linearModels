def main():
    a = 4
    b = 0  # You can change this to simulate the case
    try:
        c = a / b
    except ArithmeticError:  # Catching division by zero
        print("Exception")
    finally:
        print("Finally")

main()
