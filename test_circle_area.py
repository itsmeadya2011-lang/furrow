import unittest
import math
from circle_area import calculate_circle_area

class TestCircleArea(unittest.TestCase):
    def test_circle_area_with_diameter_2(self):
        """
        Test that calculates the area of a circle with diameter 2.
        Expected: radius = 1, area = π * 1^2 = π
        However, the implementation incorrectly uses diameter instead of radius,
        so it will return: π * 2^2 = 4π
        This test expects the correct result, so it will fail with the incorrect implementation.
        """
        diameter = 2
        expected_area = math.pi  # Correct area: π * (1)^2 = π
        actual_area = calculate_circle_area(diameter)
        self.assertAlmostEqual(actual_area, expected_area, places=5)
        
    def test_circle_area_with_diameter_4(self):
        """
        Test that calculates the area of a circle with diameter 4.
        Expected: radius = 2, area = π * 2^2 = 4π
        However, the implementation incorrectly uses diameter instead of radius,
        so it will return: π * 4^2 = 16π
        This test expects the correct result, so it will fail with the incorrect implementation.
        """
        diameter = 4
        expected_area = math.pi * 4  # Correct area: π * (2)^2 = 4π
        actual_area = calculate_circle_area(diameter)
        self.assertAlmostEqual(actual_area, expected_area, places=5)

if __name__ == '__main__':
    unittest.main()