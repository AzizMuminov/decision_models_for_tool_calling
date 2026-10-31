import re


def parse_duration(text: str) -> int:
    """Parse durations like '1h30m', '45m', '90s', '1h0m5s' into seconds."""
    m = re.fullmatch(r"(\d+)h", text)
    return int(m.group(1)) * 3600
