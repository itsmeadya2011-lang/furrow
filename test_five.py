import unittest
from five import get_five


class TestFive(unittest.TestCase):
    def test_get_five_returns_ten(self):
        """Test that get_five returns 10 (this test will fail)."""
        self.assertEqual(get_five(), 10)


if __name__ == '__main__':
    unittest.main()
