import pandas as pd
from questdb.ingress import Sender

# ======================================================
# CONFIG
# ======================================================

CHUNK_SIZE = 250_000
LOCAL_TIMEZONE = "Asia/Jerusalem"

DEVICE_SCHEMAS = {
    "TC": [
        "deviceName",
        "timestamp",
        "value1",
        "end",
    ],

    "PT": [
        "deviceName",
        "timestamp",
        "value1",
        "end",
    ],

    "RAD": [
        "deviceName",
        "timestamp",
        "value1",
        "value2",
        "value3",
        "value4",
        "value5",
        "end",
    ],

    "RH": [
        "deviceName",
        "timestamp",
        "value1",
        "value2",
        "end",
    ],

    "PRESS": [
        "deviceName",
        "timestamp",
        "value1",
        "end",
    ],

    "SONIC": [
        "deviceName",
        "timestamp",
        "value1",
        "value2",
        "value3",
        "value4",
        "end",
    ],
}


# ======================================================
# DETECT DEVICE TYPE
# ======================================================

def detect_device_type(file_path: str) -> str:
    """
    Reads only the first line of the file to determine
    the device type.
    """

    with open(file_path) as f:
        first_line = f.readline().strip()

    device_name = first_line.split(",")[0]

    # "deviceG_R 07" -> "deviceG_R"
    device_type = device_name.split("_")[0]

    return device_type


# ======================================================
# TRANSFORM
# ======================================================

def transform_chunk(df: pd.DataFrame) -> pd.DataFrame:

    # Remove XXX column
    df = df.drop(columns="end")

    # Convert timestamps to UTC
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
          .dt.tz_localize(LOCAL_TIMEZONE)
          .dt.tz_convert("UTC")
    )

    return df


# ======================================================
# QUESTDB
# ======================================================

def write_chunk(
    sender: Sender,
    table_name: str,
    df: pd.DataFrame,
):

    value_columns = [
        c for c in df.columns
        if c.startswith("value")
    ]

    for row in df.itertuples(index=False):

        columns = {
            col: getattr(row, col)
            for col in value_columns
        }

        sender.row(
            table_name=table_name,

            symbols={
                "deviceName": row.deviceName,
            },

            columns=columns,

            at=row.timestamp,
        )

    sender.flush()


# ======================================================
# IMPORT ONE FILE
# ======================================================

def import_file(
    sender: Sender,
    file_path: str,
):

    device_type = detect_device_type(file_path)

    print(f"Detected device type: {device_type}")

    schema = DEVICE_SCHEMAS[device_type]

    reader = pd.read_csv(
        file_path,
        names=schema,
        header=None,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total_rows = 0

    for chunk in reader:

        chunk = transform_chunk(chunk)

        write_chunk(
            sender,
            table_name=device_type,
            df=chunk,
        )

        total_rows += len(chunk)

        print(f"{device_type}: imported {total_rows:,} rows")


# ======================================================
# MAIN
# ======================================================

def main():
    FILE_PATH = "/data4bk/nirb/Development/guyc/May1/450_Raw_Sonic1.dat"

    with Sender.from_conf(
        "http::addr=localhost:9000;"
    ) as sender:

        import_file(sender, FILE_PATH)
    # from pathlib import Path
    #
    # directory = Path("/data4bk/nirb/Development/guyc/May1")
    #
    # for file in directory.iterdir():
    #     if file.suffix == ".dat":
    #         with Sender.from_conf(
    #             "http::addr=localhost:9000;"
    #         ) as sender:
    #
    #             import_file(sender, file)


if __name__ == "__main__":
    main()