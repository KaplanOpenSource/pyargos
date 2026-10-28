from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(
    df,
    output_root,
    station,
    device_type,
):
    if df.empty:
        return 0

    timestamp = df["timestamp"]

    groups = df.groupby(
        [
            timestamp.dt.year,
            timestamp.dt.month,
        ],
        sort=True,
    )

    total_written = 0

    for (year, month), part in groups:

        dir_name = f"{station}_{device_type}"
        output_dir = (
            Path(output_root)
            / dir_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            / f"{year:04d}-{month:02d}.parquet"
        )

        table = pa.Table.from_pandas(
            part,
            preserve_index=False,
        )

        if output_file.exists():

            existing = pq.read_table(
                output_file
            )

            table = pa.concat_tables(
                [
                    existing,
                    table,
                ],
                promote_options="default",
            )

        pq.write_table(
            table,
            output_file,
        )

        total_written += len(part)

    return total_written