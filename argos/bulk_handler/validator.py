import pandas as pd

from errors import save_errors
from config import LOCAL_TIMEZONE
from schemas import NUMERIC_COLUMNS


def validate(
    df,
    source,
    batch,
):

    # Make our own independent DataFrame
    df = df.copy()
    # -----------------------
    # deviceName
    # -----------------------

    df["deviceName"] = (
        df["deviceName"]
        .astype(str)
        .str.strip()
    )


    bad = df["deviceName"].isin(
        [
            "",
            "nan",
            "None",
            '""',
        ]
    )

    if bad.any():
        df = df.loc[~bad].copy()

        save_errors(
            df[bad],
            source,
            batch,
            "empty deviceName",
        )

        df = df[~bad]


    # -----------------------
    # timestamp
    # -----------------------

    ts = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )


    bad = ts.isna()

    if bad.any():
        df = df.loc[~bad].copy()

        save_errors(
            df[bad],
            source,
            batch,
            "bad timestamp",
        )

        df = df[~bad]

        ts = ts[~bad]


    localized = (
        ts.dt.tz_localize(
            LOCAL_TIMEZONE,
            nonexistent="NaT",
            ambiguous="NaT",
        )
    )


    bad = localized.isna()

    if bad.any():
        df = df.loc[~bad].copy()

        save_errors(
            df[bad],
            source,
            batch,
            "DST invalid timestamp",
        )

        df = df[~bad]

        localized = localized[~bad]


    df["timestamp"] = (
        localized
        .dt.tz_convert("UTC")
        .astype(
            "datetime64[ns, UTC]"
        )
    )


    # -----------------------
    # numeric validation
    # -----------------------

    for col in NUMERIC_COLUMNS:

        if col in df.columns:

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            bad = converted.isna()

            if bad.any():

                save_errors(
                    df[bad],
                    source,
                    batch,
                    f"bad numeric {col}",
                )

                df = df[~bad]

                converted = converted[~bad]

            df.loc[:, col] = converted

    return df