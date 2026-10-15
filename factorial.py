def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    
    Args:
        n: A non-negative integer
        
    Returns:
        The factorial of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Input must be non-negative")
    result = 1
    for i in range(1, n):  # BUG: Should be range(1, n+1) or range(2, n+1)
        result *= i
    return result