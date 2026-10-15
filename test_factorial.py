import unittest

from factorial import factorial


class TestFactorial(unittest.TestCase):
    
    def test_factorial_of_zero(self):
        self.assertEqual(factorial(0), 1)
        
    def test_factorial_of_one(self):
        self.assertEqual(factorial(1), 1)
        
    def test_factorial_of_two(self):
        self.assertEqual(factorial(2), 2)
        
    def test_factorial_of_five(self):
        self.assertEqual(factorial(5), 120)
        
    def test_factorial_of_ten(self):
        self.assertEqual(factorial(10), 3628800)
        
    def test_negative_input(self):
        with self.assertRaises(ValueError):
            factorial(-1)
            
    def test_negative_input_large(self):
        with self.assertRaises(ValueError):
            factorial(-10)

if __name__ == '__main__':
    unittest.main()