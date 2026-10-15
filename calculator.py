class Calculator:
    """A simple calculator supporting basic arithmetic operations."""

    def add(self, a: float, b: float) -> float:
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Return a minus b."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Return a multiplied by b."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Return a divided by b.

        Raises ZeroDivisionError if b is zero.
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b