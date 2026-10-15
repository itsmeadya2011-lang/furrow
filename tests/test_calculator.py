import pytest
from calculator import Calculator


def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0


def test_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
    assert calc.subtract(3, 5) == -2


def test_multiply():
    calc = Calculator()
    assert calc.multiply(4, 3) == 12
    assert calc.multiply(0, 5) == 0


def test_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5
    assert calc.divide(7, 2) == 3.5


def test_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)