from pathlib import Path

# ==============================
# General
# ==============================

LOCAL_TIMEZONE = "Asia/Jerusalem"

BLOCK_SIZE = 32 * 1024 * 1024   # 32 MB

# ==============================
# Output folders
# ==============================

ERROR_FOLDER = Path(
    "/home/shira/Projects/2026/argos_error_log/errors"
)

LOG_FOLDER = Path(
    "/home/shira/Projects/2026/argos_error_log/logs"
)

ERROR_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================
# QuestDB
# ==============================

QUESTDB_CONF = (
    "http::addr=localhost:9000;"
)

QUESTDB_HTTP = (
    "http://localhost:9000"
)