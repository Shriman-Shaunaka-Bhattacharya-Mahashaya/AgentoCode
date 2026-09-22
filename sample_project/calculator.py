class SimpleCalculator:
    """A simple calculator class for testing AST semantic chunking."""
    
    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Adds two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a, b):
        """Subtracts b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a, b):
        """Multiplies two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a, b):
        """Divides a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(self, a, b):
        """Raises a to the power of b (a ** b)."""
        result = a ** b
        self.history.append(f"{a} ** {b} = {result}")
        return result

    def factorial(self, n):
        """Computes the factorial of n (n!)."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers.")
        result = 1
        for i in range(2, n + 1):
            result *= i
        self.history.append(f"{n}! = {result}")
        return result

    def get_history(self):
        """Returns the calculation history."""
        return self.history