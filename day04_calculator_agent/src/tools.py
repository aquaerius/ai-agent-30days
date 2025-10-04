# src/tools.py
"""
Safe arithmetic evaluator for the Calculator tool.
Supports +, -, *, /, //, %, ** and parentheses, unary +/-, and floats/ints.
Prevents access to names, attributes, calls, or imports.
"""

import ast
import operator as op

# Allowed operators
OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

def _eval(node):
    if isinstance(node, ast.Num):              # py<3.8
        return node.n
    if isinstance(node, ast.Constant):         # numbers only
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.Expr):
        return _eval(node.value)
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")

def safe_eval(expr: str) -> float:
    """
    Evaluate a simple arithmetic expression safely.
    Raises ValueError on invalid expressions.
    """
    try:
        parsed = ast.parse(expr, mode="eval")
        return _eval(parsed.body)
    except ZeroDivisionError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def looks_like_math(text: str) -> bool:
    """
    Heuristic: if the input contains digits and math operators.
    """
    if not isinstance(text, str):
        return False
    ops = set("+-*/()%")
    has_digit = any(ch.isdigit() for ch in text)
    has_op = any(ch in ops for ch in text)
    # avoid catching sentences like "explain 2 lists"
    return has_digit and has_op
