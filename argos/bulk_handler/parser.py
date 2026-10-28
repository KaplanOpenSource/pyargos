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

# get rid of " " (for example: "WADI S" -> "WADI")
    station = match.group(1).split(' ')[0]

    raw_device_type = match.group(2)

    # get rid of numbers (for example: "SONIC1" -> "SONIC")
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