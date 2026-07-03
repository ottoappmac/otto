"""Math tool for evaluating numerical expressions."""

import re
from langchain_core.tools import BaseTool

from utilities.logger import get_logger

logger = get_logger()

# Allow only numbers, basic operators, parentheses, and spaces
_SAFE_PATTERN = re.compile(r"^[\d\s+\-*/().]+$")


def evaluate_math(expression: str) -> str:
    """Safely evaluate a math expression.

    Supports +, -, *, /, **, parentheses, and decimals.
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"
    if not _SAFE_PATTERN.match(expression):
        return "Error: only numbers and operators + - * / ** ( ) are allowed"
    try:
        result = eval(expression)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        logger.warning("Math evaluation error: %s", e)
        return f"Error: {e!s}"


class MathTool(BaseTool):
    """LangChain tool for evaluating math expressions."""

    name: str = "math"
    description: str = (
        "Evaluates a math expression. Input should be a single expression "
        "using numbers and operators: + - * / ** ( ). "
        "Example: (2 + 3) * 4 or 10 ** 2"
    )

    def _run(self, expression: str) -> str:
        logger.info("Math: %s", expression)
        print(f"[Math] expression: {expression}")
        result = evaluate_math(expression)
        logger.info("Math result: %s", result)
        print(f"[Math] result: {result}")
        return result

    async def _arun(self, expression: str) -> str:
        return self._run(expression)
