from questdb.ingress import Sender


from importer import import_file
from pathlib import Path
import  config


import json
from pathlib import Path


# ==============================
# General
# ==============================

LOCAL_TIMEZONE = "Asia/Jerusalem"

BLOCK_SIZE = 32 * 1024 * 1024   # 32 MB


# ==============================
# Device schemas
# ==============================

def load_device_schemas(path):
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    device_schemas = {
        device_type: schema["columns"]
        for device_type, schema in config["devices"].items()
    }

    non_numeric_columns = set(
        config.get("non_numeric_columns", [])
    )

    return device_schemas, non_numeric_columns


import argparse
from pathlib import Path

from config import (
    BLOCK_SIZE,
    LOCAL_TIMEZONE,
    load_device_schemas,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet."
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to a CSV file or directory containing CSV files",
    )

    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Directory where Parquet files will be written",
    )

    parser.add_argument(
        "schema_path",
        type=Path,
        help="Path to the device schema JSON file",
    )

    return parser.parse_args()

args = parse_args()

config.configure_paths(
    args.csv_path,
    args.parquet_path,
)
config.load_device_schemas(args.schema_path)

def main():


    for file in config.SOURCE_PATH:
        # try:
            import_file(
                # sender,
                file,
            )
        # except Exception as e:
        #     print()
        #     print(
        #         "FAILED:"
        #     )
        #     print(file)
        #     print(e)


if __name__ == "__main__":

    main()