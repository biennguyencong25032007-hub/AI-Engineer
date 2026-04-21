import math
from typing import Union, List


class CalculatorTool:
    """Tool for performing mathematical calculations"""

    def __init__(self):
        self.name = "calculator"
        self.description = "Perform mathematical calculations including arithmetic, algebra, and basic functions"

    def __call__(self, expression: str) -> str:
        """Evaluate a mathematical expression safely"""
        try:
            # Clean the expression
            expression = expression.strip()

            # Allowed functions and constants
            safe_dict = {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'pow': pow,
                'sqrt': math.sqrt,
                'ceil': math.ceil,
                'floor': math.floor,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e
            }

            # Evaluate in safe context
            result = eval(expression, {"__builtins__": {}}, safe_dict)

            return f"Result: {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (SyntaxError, NameError, TypeError) as e:
            return f"Error: Invalid expression - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate(self, expression: str) -> Union[float, str]:
        """Calculate and return numeric result or error message"""
        try:
            expression = expression.strip()
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
                'ceil': math.ceil, 'floor': math.floor,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def parse_and_calculate(self, query: str) -> str:
        """Parse natural language query and calculate"""
        # Extract numbers and operations from natural language
        query_lower = query.lower()

        # Simple patterns
        if 'add' in query_lower or '+' in query:
            # Extract numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                total = sum(float(n) for n in numbers)
                return f"Sum: {total}"
        elif 'subtract' in query_lower or '-' in query:
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                result = float(numbers[0]) - sum(float(n) for n in numbers[1:])
                return f"Difference: {result}"
        elif 'multiply' in query_lower or 'x' in query:
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                product = 1
                for n in numbers:
                    product *= float(n)
                return f"Product: {product}"
        elif 'divide' in query_lower or '/' in query:
            import re
            numbers = re.findall(r'\d+\.?\d*', query)
            if len(numbers) >= 2:
                try:
                    result = float(numbers[0]) / float(numbers[1])
                    return f"Quotient: {result}"
                except ZeroDivisionError:
                    return "Error: Division by zero"

        # Default: try direct evaluation
        return self.__call__(query)
