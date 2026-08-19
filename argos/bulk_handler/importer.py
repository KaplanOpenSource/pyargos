import time
import math

import pyarrow.csv as pv

from pathlib import Path

from config import BLOCK_SIZE

from schemas import DEVICE_SCHEMAS

from parser import parse_file

from validator import validate

from transform import transform

from my_questdb import (
    ensure_table,
    write,
)



def progress(
    current,
    total,
    width=40,
):

    pct = current / total

    filled = int(
        pct * width
    )

    bar = (
        "█" * filled
        +
        "-" * (width-filled)
    )

    print(
        f"\r[{bar}] "
        f"{current}/{total} "
        f"({pct:.1%})",
        end="",
    )



def import_file(
    sender,
    file_path,
):

    start = time.time()


    station, device_type, table = (
        parse_file(file_path)
    )


    schema = DEVICE_SCHEMAS[
        device_type
    ]


    ensure_table(
        table,
        schema,
    )


    # -------------------------
    # Estimate number of batches
    # -------------------------

    file_size = (
        Path(file_path)
        .stat()
        .st_size
    )


    total_batches = max(
        1,
        math.ceil(
            file_size /
            BLOCK_SIZE
        )
    )


    print()
    print("="*60)
    print(file_path)
    print("="*60)


    reader = pv.open_csv(
        file_path,

        read_options=pv.ReadOptions(
            column_names=schema,
            block_size=BLOCK_SIZE,
        ),
    )


    total_read = 0
    total_written = 0
    total_skipped = 0


    for batch_no, batch in enumerate(
        reader,
        start=1,
    ):


        df = batch.to_pandas()


        total_read += len(df)


        before = len(df)


        df["station"] = station

        df["deviceType"] = device_type



        df = validate(
            df,
            file_path,
            batch_no,
        )


        after_validation = len(df)


        total_skipped += (
            before -
            after_validation
        )


        df = transform(df)

        try:

            written = write(
                sender,
                table,
                df,
            )

            total_written += written

        except Exception as e:

            print(
                f"\nERROR writing batch "
                f"{batch_no}: {e}"
            )

            print(
                "Skipping this batch and "
                "continuing with the next one."
            )

            continue



        progress(
            batch_no,
            total_batches,
        )



    elapsed = (
        time.time()
        -
        start
    )


    print()

    print("-"*60)

    print(
        f"Finished {file_path}"
    )

    print(
        f"Read:    {total_read:,}"
    )

    print(
        f"Written: {total_written:,}"
    )

    print(
        f"Skipped: {total_skipped:,}"
    )

    print(
        f"Time:    {elapsed/60:.1f} minutes"
    )

    print("-"*60)