# tests/test_tools.py
import pytest
from src.tools import safe_eval, looks_like_math

def test_basic_arithmetic():
    assert safe_eval("17*42") == 714
    assert safe_eval("100+250") == 350
    assert safe_eval("(2+3)*4") == 20
    assert safe_eval("2**5") == 32
    assert safe_eval("7//3") == 2
    assert safe_eval("7%3") == 1

def test_unary_ops():
    assert safe_eval("-5 + +2") == -3

def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        safe_eval("1/0")

def test_disallowed_names():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hacked')")

def test_looks_like_math():
    assert looks_like_math("What is 2+2?")
    assert looks_like_math("100 * 3")
    assert not looks_like_math("Explain lists in Python")
