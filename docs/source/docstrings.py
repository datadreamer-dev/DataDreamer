def append_new_lines(lines, n=1):
    for _ in range(n):
        lines.append("")


def clear(lines):
    lines.clear()
    append_new_lines(lines, 1)


def split_lines(v):
    return v.strip().split("\n")


def indent_lines(lines):
    lines.copy()
    for line_idx in range(len(lines)):
        lines[line_idx] = f"    {lines[line_idx]}"
    return lines


def append(lines, v, indent_level=0):
    v = split_lines(v)
    for _ in range(indent_level):
        v = indent_lines(v)
    lines.extend(v)


STEP_HELP = """.. code-block:: jsonnet
    :caption: CLS_NAME.help
"""

