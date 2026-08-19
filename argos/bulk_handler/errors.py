from pathlib import Path
import pandas as pd

from config import ERROR_FOLDER


def save_errors(
    df,
    source_file,
    batch,
    reason,
):

    if df.empty:
        return

    out = df.copy()

    out.insert(
        0,
        "reason",
        reason,
    )

    out.insert(
        0,
        "batch",
        batch,
    )

    out.insert(
        0,
        "source_file",
        Path(source_file).name,
    )

    out["timestamp"] = (
        out["timestamp"]
        .astype(str)
    )

    target = (
        ERROR_FOLDER /
        (
            Path(source_file).name
            +
            ".errors.csv"
        )
    )

    out.to_csv(
        target,
        mode="a",
        header=not target.exists(),
        index=False,
    )