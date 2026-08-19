from pathlib import Path
import re


def parse_file(path):

    filename = Path(path).stem

    match = re.match(
        r"^(.*?)_Raw_(.+)$",
        filename,
    )

    if not match:
        raise ValueError(
            f"Bad filename: {filename}"
        )

    station = match.group(1)

    raw_device_type = match.group(2)

    device_type = re.sub(
        r"\d+$",
        "",
        raw_device_type,
    )

    table_name = (
        f"{station}_{raw_device_type}"
    )

    return (
        station,
        device_type,
        table_name,
    )