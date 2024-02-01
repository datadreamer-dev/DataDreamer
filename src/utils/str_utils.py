import re
from itertools import chain


def replace_many(text, substitions):
    pattern = re.compile("|".join(map(re.escape, substitions.keys())))
    return pattern.sub(lambda match: substitions[match.group(0)], text)


def get_templated_var_names(templated_str):
    escaped_components = re.split(r"{{|}}", templated_str)  # Ignore {{ and }}
    template_var_pattern = r"{([a-zA-Z0-9:_\.]+)}"
    return list(
        chain.from_iterable(
            [
                re.findall(template_var_pattern, component)
                for component in escaped_components
            ]
        )
    )


def replace_templated_vars(templated_str, var_name_to_values):
    return replace_many(
        templated_str, {"{" + k + "}": str(v) for k, v in var_name_to_values.items()}
    )
