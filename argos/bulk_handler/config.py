from pathlib import Path
import json


LOCAL_TIMEZONE = "Asia/Jerusalem"
BLOCK_SIZE = 32 * 1024 * 1024


# ==============================
# Runtime configuration
# ==============================

SOURCE_PATH = None
PARQUET_ROOT = None
ERROR_FOLDER = None
LOG_FOLDER = None

DEVICE_SCHEMAS = {}
NON_NUMERIC_COLUMNS = set()


def configure_paths(csv_path, parquet_path):
    global SOURCE_PATH
    global PARQUET_ROOT
    global ERROR_FOLDER
    global LOG_FOLDER

    if csv_path.is_dir():
        SOURCE_PATH = csv_path.rglob('*.dat')
    elif csv_path.is_file():
        SOURCE_PATH = [csv_path]
    print(type(SOURCE_PATH))
    PARQUET_ROOT = Path(parquet_path)

    ERROR_FOLDER = PARQUET_ROOT / "errors"
    LOG_FOLDER = PARQUET_ROOT / "logs"

    ERROR_FOLDER.mkdir(parents=True, exist_ok=True)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)


def load_device_schemas(path):
    global DEVICE_SCHEMAS
    global NON_NUMERIC_COLUMNS

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    print("Loaded devices:")
    print(config.keys())
    DEVICE_SCHEMAS = config["devices"]

    #
    # NON_NUMERIC_COLUMNS = set(
    #     config.get("non_numeric_columns", [])
    # )
# # ==============================
# # QuestDB
# # ==============================
#
# QUESTDB_CONF = (
#     "http::addr=localhost:9000;"
# )
#
# QUESTDB_HTTP = (
#     "http://localhost:9000"
# )
