import ast
import javalang


def java_code_percentage(text: str) -> float:
    """
    Calculate the percentage of characters in the input text
    that form valid Java code snippets.

    Args:
        text (str): The input string.
    Returns:
        float: Percentage of characters that are valid Java code.
    """
    if not text.strip():
        return 0.0

    total_chars = len(text)
    valid_chars = 0

    # Split into potential Java code blocks by semicolons and newlines
    chunks = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]
    for chunk in chunks:
        try:
            # Try parsing the chunk using javalang
            javalang.parse.parse(chunk)
            valid_chars += len(chunk)
        except:
            pass  # Ignore invalid Java chunks

    return valid_chars / total_chars


def python_code_percentage(text: str) -> float:
    """
    Calculate the percentage of characters in the input text
    that form valid Python code snippets.

    Args:
        text (str): The input string.
    Returns:
        float: Percentage of characters that are valid Python code.
    """
    if not text.strip():
        return 0.0

    total_chars = len(text)
    valid_chars = 0

    # Split into potential Python code blocks by semicolons and newlines
    chunks = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]
    for chunk in chunks:
        try:
            ast.parse(chunk)  # Try parsing the chunk
            valid_chars += len(chunk)
        except SyntaxError:
            pass  # Ignore invalid code chunks

    return valid_chars / total_chars
