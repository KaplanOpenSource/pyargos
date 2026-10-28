import time
import requests

# ============================================================
# CONFIG
# ============================================================

QUESTDB_URL = "http://localhost:9000"

# Relative to QuestDB's export root.
#
# Default:
# <questdb_root>/export/
#
PARQUET_FOLDER = "parquet"


# Export everything from this date onward.
#
# Examples:
#
# "2025-05-01"  -> May 2025 onward
# "2026-01-01"  -> January 2026 onward
#
START_MONTH = "2026-06-01"

# How often to check export status
POLL_SECONDS = 5


# ============================================================
# QUESTDB SQL
# ============================================================

def execute_sql(query):
    """
    Execute SQL through QuestDB's REST API.
    """

    response = requests.get(
        f"{QUESTDB_URL}/exec",
        params={"query": query},
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data


# ============================================================
# GET TABLES
# ============================================================

def get_tables():
    """
    Return all user tables that have a designated timestamp.

    Tables without a designated timestamp cannot be exported
    using PARTITION_BY MONTH.
    """

    query = """
    SELECT
        table_name,
        designatedTimestamp
    FROM tables()
    WHERE table_name NOT LIKE 'sys.%'
    ORDER BY table_name
    """

    data = execute_sql(query)

    tables = []

    for row in data["dataset"]:
        table_name = row[0]
        timestamp_column = row[1]

        tables.append({
            "name": table_name,
            "timestamp": timestamp_column,
        })

    return tables

def get_specific_tables():
    """
    Return all AB_* tables.
    """

    query = """
    SELECT table_name
    FROM tables()
    WHERE table_name LIKE 'WADI_%'
    ORDER BY table_name
    """

    data = execute_sql(query)

    return [
        row[0]
        for row in data["dataset"]
    ]
# ============================================================
# START EXPORT
# ============================================================

def start_export(table_name):#,timestamp_column):
    """
    Start asynchronous QuestDB COPY TO Parquet export.

    The Parquet output is partitioned by MONTH.
    """

    destination = f"{PARQUET_FOLDER}/{table_name}"

    # query = f"""
    # COPY (
    #     SELECT *
    #     FROM "{table_name}"
    #     WHERE "{timestamp_column}" >= '{START_MONTH}'
    # )
    # TO '{destination}'
    # WITH
    #     FORMAT PARQUET
    #     PARTITION_BY MONTH
    # """
    query = f"""
    COPY (
        SELECT *
        FROM "{table_name}"
    )
    TO '{destination}'
    WITH
        FORMAT PARQUET
        PARTITION_BY MONTH
    """

    data = execute_sql(query)

    # COPY TO returns the export ID
    export_id = data["dataset"][0][0]

    return export_id


# ============================================================
# CHECK EXPORT STATUS
# ============================================================

def get_export_status(export_id):

    query = f"""
    SELECT
        id,
        table_name,
        export_path,
        num_exported_files,
        phase,
        status,
        message,
        errors
    FROM sys.copy_export_log
    WHERE id = '{export_id}'
    ORDER BY ts DESC
    """

    data = execute_sql(query)

    if not data["dataset"]:
        return None

    columns = [
        col["name"]
        for col in data["columns"]
    ]

    row = data["dataset"][0]

    return dict(zip(columns, row))


# ============================================================
# WAIT FOR EXPORT
# ============================================================

def wait_for_export(table_name, export_id):

    print()
    print(f"[{table_name}]")
    print(f"Export ID: {export_id}")

    while True:

        status = get_export_status(export_id)

        if status is None:

            print(
                "Waiting for QuestDB "
                "to start export..."
            )

            time.sleep(POLL_SECONDS)
            continue

        phase = status["phase"]
        state = status["status"]
        files = status["num_exported_files"]

        print(
            f"\r"
            f"phase={phase} | "
            f"status={state} | "
            f"files={files}",
            end="",
        )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        if state == "finished":

            print()

            print(
                f"✓ {table_name} finished "
                f"({files} monthly Parquet files)"
            )

            return True

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        if state == "failed":

            print()

            print(
                f"✗ {table_name} FAILED"
            )

            print(
                f"Message: {status['message']}"
            )

            print(
                f"Errors: {status['errors']}"
            )

            return False

        # ----------------------------------------------------
        # Cancelled
        # ----------------------------------------------------

        if state == "cancelled":

            print()

            print(
                f"✗ {table_name} CANCELLED"
            )

            return False

        time.sleep(POLL_SECONDS)


# ============================================================
# EXPORT ONE TABLE
# ============================================================

def export_table(table):

    table_name = table["name"]
    timestamp_column = table["timestamp"]

    print()
    print("=" * 70)
    print(f"Starting export: {table_name}")
    print("=" * 70)

    print(
        f"Timestamp column: {timestamp_column}"
    )

    # --------------------------------------------------------
    # Tables without a designated timestamp
    # --------------------------------------------------------

    if timestamp_column is None:

        print(
            f"⚠ {table_name} has no designated timestamp."
        )

        print(
            "Skipping because monthly partitioning "
            "requires a designated timestamp."
        )

        return False

    try:

        export_id = start_export(
            table_name,
            timestamp_column
        )

        return wait_for_export(
            table_name,
            export_id,
        )

    except Exception as e:

        print()

        print(
            f"✗ {table_name} FAILED"
        )

        print(e)

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    # print()
    # print("=" * 70)
    # print("QUESTDB → MONTHLY PARQUET")
    # print("=" * 70)
    #
    # # --------------------------------------------------------
    # # Get tables
    # # --------------------------------------------------------
    #
    # tables = get_tables()
    #
    # print()
    # print(
    #     f"Found {len(tables)} tables:"
    # )
    #
    # for table in tables:
    #
    #     timestamp = table["timestamp"]
    #
    #     if timestamp:
    #
    #         print(
    #             f"  - {table['name']} "
    #             f"(timestamp: {timestamp})"
    #         )
    #
    #     else:
    #
    #         print(
    #             f"  - {table['name']} "
    #             f"(NO designated timestamp)"
    #         )
    #
    # print()
    #
    # print(
    #     f"Output folder: {PARQUET_FOLDER}"
    # )
    #
    # print(
    #     "Partitioning: MONTH"
    # )
    #
    # # --------------------------------------------------------
    # # Export tables
    # # --------------------------------------------------------
    # timestamp_column = table["timestamp"]
    # successful = []
    # failed = []
    #
    # for table in tables:
    #
    #     success = export_table(table)
    #
    #     if success:
    #
    #         successful.append(
    #             table["name"]
    #         )
    #
    #     else:
    #
    #         failed.append(
    #             table["name"]
    #         )

    # WADI S_Sonic1
    # WADI S_Sonic2
    table_name = "WADI S_Sonic2"

    export_id = start_export(table_name)

    wait_for_export(
        table_name,
        export_id,
    )
    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    # print()
    # print("=" * 70)
    # print("EXPORT SUMMARY")
    # print("=" * 70)
    #
    # print(
    #     f"Total tables: {len(tables)}"
    # )
    #
    # print(
    #     f"Successful:   {len(successful)}"
    # )
    #
    # print(
    #     f"Failed:       {len(failed)}"
    # )
    #
    # if successful:
    #
    #     print()
    #     print("Successful tables:")
    #
    #     for table in successful:
    #
    #         print(
    #             f"  ✓ {table}"
    #         )
    #
    # if failed:
    #
    #     print()
    #     print("Failed tables:")
    #
    #     for table in failed:
    #
    #         print(
    #             f"  ✗ {table}"
    #         )
    #
    # print()
    # print("Done.")


if __name__ == "__main__":
    main()


#
#
# import time
# import requests
#
# # ============================================================
# # CONFIG
# # ============================================================
#
# QUESTDB_URL = "http://localhost:9000"
#
# # QuestDB COPY export root.
# #
# # If cairo.sql.copy.export.root is configured in server.conf,
# # this path is relative to that root.
# #
# PARQUET_FOLDER = "parquet"
#

#
# # ------------------------------------------------------------
# # EXPORT START DATE
# # ------------------------------------------------------------
# #
# # Export everything from this date onward.
# #
# # Examples:
# #
# # "2025-05-01"  -> May 2025 onward
# # "2026-01-01"  -> January 2026 onward
# #
# START_MONTH = "2026-06-01"
#
# # How often to check export status
# POLL_SECONDS = 5
#
#
# # ============================================================
# # QUESTDB SQL
# # ============================================================
#
# def execute_sql(query):
#     """
#     Execute SQL through QuestDB's REST API.
#     """
#
#     response = requests.get(
#         f"{QUESTDB_URL}/exec",
#         params={"query": query},
#         timeout=60,
#     )
#
#     response.raise_for_status()
#
#     data = response.json()
#
#     if "error" in data:
#         raise RuntimeError(data["error"])
#
#     return data
#
#
# # ============================================================
# # GET TABLES
# # ============================================================
#
# def get_tables():
#     """
#     Return all user tables and their designated timestamp.
#     """
#
#     query = """
#     SELECT
#         table_name,
#         designatedTimestamp
#     FROM tables()
#     WHERE table_name NOT LIKE 'sys.%'
#     ORDER BY table_name
#     """
#
#     data = execute_sql(query)
#
#     tables = []
#
#     for row in data["dataset"]:
#
#         table_name = row[0]
#         timestamp_column = row[1]
#
#         tables.append({
#             "name": table_name,
#             "timestamp": timestamp_column,
#         })
#
#     return tables
#
#
# # ============================================================
# # START EXPORT
# # ============================================================
#
# def start_export(table_name, timestamp_column):
#     """
#     Start an asynchronous QuestDB Parquet export.
#
#     Only rows from START_MONTH onward are exported.
#     Output is partitioned by month.
#     """
#
#     destination = (
#         f"{PARQUET_FOLDER}/{table_name}"
#     )
#
#     query = f"""
#     COPY (
#         SELECT *
#         FROM "{table_name}"
#         WHERE "{timestamp_column}" >= '{START_MONTH}'
#     )
#     TO '{destination}'
#     WITH
#         FORMAT PARQUET
#         PARTITION_BY MONTH
#     """
#
#     data = execute_sql(query)
#
#     # COPY TO returns the export ID
#     export_id = data["dataset"][0][0]
#
#     return export_id
#
#
# # ============================================================
# # CHECK EXPORT STATUS
# # ============================================================
#
# def get_export_status(export_id):
#
#     query = f"""
#     SELECT
#         id,
#         table_name,
#         export_path,
#         num_exported_files,
#         phase,
#         status,
#         message,
#         errors
#     FROM sys.copy_export_log
#     WHERE id = '{export_id}'
#     ORDER BY ts DESC
#     """
#
#     data = execute_sql(query)
#
#     if not data["dataset"]:
#         return None
#
#     columns = [
#         col["name"]
#         for col in data["columns"]
#     ]
#
#     row = data["dataset"][0]
#
#     return dict(
#         zip(columns, row)
#     )
#
#
# # ============================================================
# # WAIT FOR EXPORT
# # ============================================================
#
# def wait_for_export(
#     table_name,
#     export_id,
# ):
#
#     print()
#     print(
#         f"[{table_name}]"
#     )
#
#     print(
#         f"Export ID: {export_id}"
#     )
#
#     while True:
#
#         status = get_export_status(
#             export_id
#         )
#
#         if status is None:
#
#             print(
#                 "Waiting for QuestDB "
#                 "to start export..."
#             )
#
#             time.sleep(
#                 POLL_SECONDS
#             )
#
#             continue
#
#         phase = status["phase"]
#         state = status["status"]
#         files = status[
#             "num_exported_files"
#         ]
#
#         print(
#             f"\r"
#             f"phase={phase} | "
#             f"status={state} | "
#             f"files={files}",
#             end="",
#         )
#
#         # ----------------------------------------------------
#         # FINISHED
#         # ----------------------------------------------------
#
#         if state == "finished":
#
#             print()
#
#             print(
#                 f"✓ {table_name} finished "
#                 f"({files} Parquet files)"
#             )
#
#             return True
#
#         # ----------------------------------------------------
#         # FAILED
#         # ----------------------------------------------------
#
#         if state == "failed":
#
#             print()
#
#             print(
#                 f"✗ {table_name} FAILED"
#             )
#
#             print(
#                 f"Message: "
#                 f"{status['message']}"
#             )
#
#             print(
#                 f"Errors: "
#                 f"{status['errors']}"
#             )
#
#             return False
#
#         # ----------------------------------------------------
#         # CANCELLED
#         # ----------------------------------------------------
#
#         if state == "cancelled":
#
#             print()
#
#             print(
#                 f"✗ {table_name} CANCELLED"
#             )
#
#             return False
#
#         time.sleep(
#             POLL_SECONDS
#         )
#
#
# # ============================================================
# # EXPORT ONE TABLE
# # ============================================================
#
# def export_table(table):
#
#     table_name = table["name"]
#     timestamp_column = table["timestamp"]
#
#     print()
#     print("=" * 70)
#     print(
#         f"Starting export: {table_name}"
#     )
#     print("=" * 70)
#
#     print(
#         f"Timestamp column: "
#         f"{timestamp_column}"
#     )
#
#     print(
#         f"Starting from: "
#         f"{START_MONTH}"
#     )
#
#     # --------------------------------------------------------
#     # No timestamp
#     # --------------------------------------------------------
#
#     if timestamp_column is None:
#
#         print(
#             f"⚠ {table_name} has no "
#             f"designated timestamp."
#         )
#
#         print(
#             "Skipping."
#         )
#
#         return False
#
#     try:
#
#         export_id = start_export(
#             table_name,
#             timestamp_column,
#         )
#
#         return wait_for_export(
#             table_name,
#             export_id,
#         )
#
#     except Exception as e:
#
#         print()
#
#         print(
#             f"✗ {table_name} FAILED"
#         )
#
#         print(e)
#
#         return False
#
#
# # ============================================================
# # MAIN
# # ============================================================
#
# def main():
#
#     print()
#     print("=" * 70)
#     print("QUESTDB → MONTHLY PARQUET EXPORT")
#     print("=" * 70)
#
#     print()
#     print(
#         f"Starting from: {START_MONTH}"
#     )
#
#     print(
#         f"Output folder: {PARQUET_FOLDER}"
#     )
#
#     print(
#         "Partitioning: MONTH"
#     )
#
#     # --------------------------------------------------------
#     # Get tables
#     # --------------------------------------------------------
#
#     tables = get_tables()
#
#     print()
#     print(
#         f"Found {len(tables)} tables:"
#     )
#
#     for table in tables:
#
#         timestamp = table["timestamp"]
#
#         if timestamp:
#
#             print(
#                 f"  - {table['name']} "
#                 f"(timestamp: {timestamp})"
#             )
#
#         else:
#
#             print(
#                 f"  - {table['name']} "
#                 f"(NO designated timestamp)"
#             )
#
#     # --------------------------------------------------------
#     # Export
#     # --------------------------------------------------------
#
#     successful = []
#     failed = []
#
#     for table in tables:
#
#         success = export_table(
#             table
#         )
#
#         if success:
#
#             successful.append(
#                 table["name"]
#             )
#
#         else:
#
#             failed.append(
#                 table["name"]
#             )
#
#     # --------------------------------------------------------
#     # Summary
#     # --------------------------------------------------------
#
#     print()
#     print("=" * 70)
#     print("EXPORT SUMMARY")
#     print("=" * 70)
#
#     print(
#         f"Start month: {START_MONTH}"
#     )
#
#     print(
#         f"Total tables: {len(tables)}"
#     )
#
#     print(
#         f"Successful:   {len(successful)}"
#     )
#
#     print(
#         f"Failed:       {len(failed)}"
#     )
#
#     if successful:
#
#         print()
#         print("Successful:")
#
#         for table in successful:
#
#             print(
#                 f"  ✓ {table}"
#             )
#
#     if failed:
#
#         print()
#         print("Failed:")
#
#         for table in failed:
#
#             print(
#                 f"  ✗ {table}"
#             )
#
#     print()
#     print("Done.")
#
#
# # ============================================================
# # RUN
# # ============================================================
#
# if __name__ == "__main__":
#     main()