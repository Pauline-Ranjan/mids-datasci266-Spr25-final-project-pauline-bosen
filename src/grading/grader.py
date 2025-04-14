"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call: 
grade_answer(given_answer: str, ground_truth: str)
"""
import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from sentence_transformers import SentenceTransformer, util

from grading import math_normalize


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def extract_final_answer(text):
    boxed_matches = re.findall(r"\\boxed{([^}]+)}", text)
    if boxed_matches:
        for i in range(len(boxed_matches)-1):
            if boxed_matches[i].strip() != 'answer':
                return boxed_matches[i].strip()
    match = re.search(r"####\s*([\w\-\+\*/\^\\\(\)\.,]+)", text)
    if match:
        return match.group(1).strip()
    if "The final answer is:" in text:
        text = text.split("The final answer is:")[-1]
    inline_matches = re.findall(r"\$([^$]+)\$", text)
    if inline_matches:
        return inline_matches[-1].strip()

    list_match = re.findall(r"\[[^\[\]]+\]", text)
    if list_match:
        return list_match[-1].strip()

    number_match = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if number_match:
        return number_match[-1].strip()

    return text.strip()

def extract_final_ground_truth(text: str) -> str:
    """
    Extract a final answer from the ground truth.
    Handles:
    - #### VAL
    - <<...=VAL>>VAL
    - final math expression
    - final fallback string
    """
    # Case 1: Markdown-style
    match = re.search(r"####\s*([\w\-\+\*/\^\\\(\)\.,]+)", text)
    if match:
        return match.group(1).strip()
    # Case 2: Annotated pattern
    match = re.findall(r"<<[^=]*=([^>]+)>>\s*([\w\-\+\*/\^\\\(\)\.,]+)", text)
    if match:
        return match[-1][1].strip()
    # Case 3: Final math expression with = (e.g., "x = 5")
    match = re.findall(r"= *([\w\-\+\*/\^\\\(\)\.,]+)", text)
    if match:
        return match[-1].strip()

    # Case 4: Final fallback: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def grade_answer_with_label(response: str, truth: str) -> str:
    """
    Return an accuracy label indicating the type of match:
    - "correct_mathd": passed mathd normalization check
    - "correct_string": normalized strings match
    - "correct_symbolic": passed symbolic equivalence check
    - "correct_semantic": semantically similar via transformer embeddings
    - "incorrect": no match
    """
    given_answer = extract_final_answer(response)
    ground_truth = extract_final_ground_truth(truth)


    if given_answer is None:
        return "incorrect"

    # (1) mathd normalized match
    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return "correct_mathd"

    # (2) manual normalization + exact string match
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized and ground_truth_normalized == given_normalized:
        return "correct_string"

    # (3) symbolic math match via sympy
    symbolic_match = False
    if ground_truth_normalized and given_normalized:
        ground_truth_elems = split_tuple(ground_truth_normalized)
        given_elems = split_tuple(given_normalized)

        if not (
            len(ground_truth_elems) > 1 and (
                ground_truth_normalized[0] != given_normalized[0]
                or ground_truth_normalized[-1] != given_normalized[-1]
            )
        ) and len(ground_truth_elems) == len(given_elems):

            symbolic_match = True
            for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
                if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                    if ground_truth_elem != given_elem:
                        symbolic_match = False
                        break
                elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                    symbolic_match = False
                    break
                elif not are_equal_under_sympy(ground_truth_elem, given_elem):
                    symbolic_match = False
                    break

    if symbolic_match:
        return "correct_symbolic"

    # (4) semantic similarity
    if semantic_similarity(given_answer, ground_truth) > 0.85:
        return "correct_semantic"

    return "incorrect"


_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(response: str, solution: str) -> bool: 
    """
    Evaluate Semantic Similarity with SentenceTransformer
    """
    pred_emb = _model.encode(response, convert_to_tensor=True)
    ans_emb = _model.encode(solution, convert_to_tensor=True)

    return util.cos_sim(pred_emb, ans_emb).item()
