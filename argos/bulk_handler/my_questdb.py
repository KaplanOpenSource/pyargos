import requests
from questdb.ingress import Sender

from config import QUESTDB_HTTP
from schemas import NUMERIC_COLUMNS


def sql(query):

    response = requests.get(
        f"{QUESTDB_HTTP}/exec",
        params={"query": query},
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result


def ensure_table(table_name, schema):

    columns = [
        '"deviceName" SYMBOL'
    ]

    for col in schema:

        if col in {
            "deviceName",
            "timestamp",
            "end",
        }:
            continue

        if col in NUMERIC_COLUMNS:
            columns.append(
                f'"{col}" DOUBLE'
            )

    columns.extend([
        '"station" VARCHAR',
        '"deviceType" VARCHAR',
        '"timestamp" TIMESTAMP_NS',
    ])

    query = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        {",".join(columns)}
    )
    TIMESTAMP(timestamp)
    PARTITION BY DAY
    WAL
    DEDUP UPSERT KEYS(timestamp, deviceName);
    """

    sql(query)

def write(sender, table, df):

    if df.empty:
        return 0

    df = df.copy()

    # --------------------------------------------
    # Normalize deviceName
    # --------------------------------------------

    df["deviceName"] = (
        df["deviceName"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------
    # Find bad deviceName rows
    # --------------------------------------------

    bad = (
        df["deviceName"].eq("")
        | df["deviceName"].isin([
            "nan",
            "NaN",
            "None",
            '""',
        ])
    )

    if bad.any():

        print()
        print(
            f"!!! FOUND {bad.sum()} BAD deviceName ROW(S) !!!"
        )

        print(
            df.loc[
                bad,
                ["deviceName", "timestamp"]
            ].head(20)
        )

        # Remove them
        df = df.loc[~bad].copy()

    # --------------------------------------------
    # Nothing left
    # --------------------------------------------

    if df.empty:
        print(
            "Entire batch contained invalid "
            "deviceName rows. Skipping."
        )
        return 0

    # --------------------------------------------
    # Send
    # --------------------------------------------

    sender.dataframe(
        table_name=table,
        df=df,
        symbols=["deviceName"],
        at="timestamp",
    )

    sender.flush()

    return len(df)