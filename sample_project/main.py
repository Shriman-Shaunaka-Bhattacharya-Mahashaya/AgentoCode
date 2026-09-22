from calculator import SimpleCalculator

def main():
    print("Welcome to the Sample Calculator App")
    calc = SimpleCalculator()
    
    print(f"10 + 5 = {calc.add(10, 5)}")
    print(f"20 - 7 = {calc.subtract(20, 7)}")
    print(f"3 * 4 = {calc.multiply(3, 4)}")
    # Test power function
    print(f"2 ** 8 = {calc.power(2, 8)}")
    
    # Test factorial function
    print(f"5! = {calc.factorial(5)}")
    print(f"7! = {calc.factorial(7)}")
    # Additional call with input 3
    print(f"3! = {calc.factorial(3)}")
    
    try:
        print(f"15 / 3 = {calc.divide(15, 3)}")
        print(f"5 / 0 = {calc.divide(5, 0)}")
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\nHistory:")
    for record in calc.get_history():
        print(f"- {record}")

if __name__ == "__main__":
    main()
