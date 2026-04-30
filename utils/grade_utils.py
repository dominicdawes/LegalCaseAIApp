# utils/grade_utils.py

def calculate_letter_grade(percentage: int) -> str:
    """
    Deterministic mapping of percentage to Letter Grade.
    Standard Law School Curve approximation.
    """
    if percentage >= 97: return "A+"
    if percentage >= 93: return "A"
    if percentage >= 89: return "A-"
    if percentage >= 86: return "B+"
    if percentage >= 83: return "B"
    if percentage >= 80: return "B-"
    if percentage >= 76: return "C+"
    if percentage >= 73: return "C"
    if percentage >= 70: return "C-"
    if percentage >= 60: return "D"
    return "F"