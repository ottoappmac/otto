from typing import Dict, Any


def convert_dict_to_string(data: Dict[str, Any], indent: int = 2) -> str:
    lines = []
    indent_str = ' ' * indent

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(convert_dict_to_string(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}: [")
            for item in value:
                if isinstance(item, dict):
                    lines.append(convert_dict_to_string(item, indent + 2))
                else:
                    lines.append(f"{' ' * (indent + 2)}{item}")
            lines.append(f"{indent_str}]")
        else:
            lines.append(f"{indent_str}{key}: {value}")
    return '\n'.join(lines)
