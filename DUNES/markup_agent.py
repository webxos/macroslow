# markup_agent.py: MARKUP Agent for reverse Markdown generation
# Purpose: Creates .mu files for error detection and auditability
# Instructions:
# 1. Input is any Markdown content
# 2. Output reverses both structure and text (e.g., "Hello" -> "olleH")
def generate_mu_file(content: str) -> str:
    """
    Generate a .mu file by reversing Markdown structure and content.
    Args:
        content (str): Input Markdown content
    Returns:
        str: Reversed .mu content
    """
    # Split content into lines and reverse structure
    lines = content.split("\n")
    reversed_lines = [line[::-1] for line in lines]
    return "\n".join(reversed_lines[::-1])