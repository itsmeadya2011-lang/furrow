import pytest
from calculator import Calculator


def test_add_comprehensive():
    calc = Calculator()
    assert calc.add(0, 0) == 0
    assert calc.add(1.5, 2.5) == 4.0
    assert calc.add(-10, -5) == -15
    assert calc.add(float('inf'), 1) == float('inf')
    assert calc.add(-float('inf'), -1) == -float('inf')


def test_subtract_comprehensive():
    calc = Calculator()
    assert calc.subtract(10, 10) == 0
    assert calc.subtract(1.5, 0.5) == 1.0
    assert calc.subtract(-5, -3) == -2
    assert calc.subtract(0, 0) == 0


def test_multiply_comprehensive():
    calc = Calculator()
    assert calc.multiply(1, 1) == 1
    assert calc.multiply(2.5, 4) == 10.0
    assert calc.multiply(-3, 4) == -12
    assert calc.multiply(-3, -4) == 12
    assert calc.multiply(10**6, 10**6) == 10**12


def test_divide_comprehensive():
    calc = Calculator()
    assert calc.divide(5, 1) == 5
    assert calc.divide(1, 2) == 0.5
    assert calc.divide(0, 5) == 0
    assert calc.divide(-10, 2) == -5
    assert calc.divide(10, -2) == -5
    assert calc.divide(-10, -2) == 5


def test_divide_by_zero_comprehensive():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError) as exc_info:
        calc.divide(5, 0)
    assert str(exc_info.value) == "Cannot divide by zero"
