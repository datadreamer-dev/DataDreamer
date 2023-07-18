def safe_fn(value: str):
    value = value.replace(" ", "-")
    safe_chars = ("-", "_", "/")
    return "".join(c for c in value if c.isalnum() or c in safe_chars).strip()
