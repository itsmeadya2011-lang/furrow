import math

def calculate_circle_area(diameter):
    """
    Calculate the area of a circle given its diameter.
    Intentionally incorrect implementation: uses diameter instead of radius.
    
    Correct formula: area = π * (radius)^2 where radius = diameter/2
    Incorrect formula: area = π * (diameter)^2
    """
    return math.pi * (diameter ** 2)