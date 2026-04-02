"""
Dask-based interface to Cassandra time-series data.

Provides the ``CassandraBag`` class for querying device telemetry from
a Cassandra database (typically the ThingsBoard backend) using Dask
for parallel, partitioned reads.

Performance notes:

- Uses a single shared Cassandra session (connection pooling) instead of
  opening/closing a connection per partition.
- Queries all metric keys in a single CQL query with ``key IN (...)``
  instead of looping per key.
- Uses ``dask.delayed`` → ``pd.DataFrame`` instead of ``dask.bag`` to
  avoid the bag→DataFrame→pivot materialization overhead.
- Uses ``extend()`` instead of list concatenation to avoid O(n^2).
"""

from cassandra.cluster import Cluster
import dask
import dask.dataframe as dd
import pandas
import numpy as np


class CassandraBag:
    """
    Dask interface for querying Cassandra time-series data.

    Designed for querying the ThingsBoard ``ts_kv_cf`` table, which stores
    device telemetry as key-value pairs partitioned by month.

    Uses a single shared Cassandra session for all queries (created on
    init, closed on ``close()`` or garbage collection).

    Parameters
    ----------
    deviceID : str
        The UUID of the device to query.
    IP : str, optional
        The Cassandra node IP address. Defaults to ``"127.0.0.1"``.
    db_name : str, optional
        The database (keyspace) name. Defaults to ``"thingsboard"``.
    set_name : str, optional
        The table name. Defaults to ``"ts_kv_cf"``.
    fetch_size : int, optional
        Number of rows per Cassandra fetch page. Higher values reduce
        round-trips but use more memory. Defaults to 50000.

    Examples
    --------
    >>> bag = CassandraBag(deviceID="727b0e40-5b96-11e9-989b-eb5e36f2a0b8")
    >>> df = bag.getDataFrame("2024-01-01", "2024-01-31")
    >>> bag.close()
    """

    def __init__(self, deviceID, IP='127.0.0.1', db_name='thingsboard',
                 set_name='ts_kv_cf', fetch_size=50000):
        """
        Initialize CassandraBag with a persistent session.

        Creates a Cassandra cluster connection and session that is reused
        across all queries. Call ``close()`` when done to release resources.

        Parameters
        ----------
        deviceID : str
            The UUID of the device to query.
        IP : str, optional
            The Cassandra node IP address. Defaults to ``"127.0.0.1"``.
        db_name : str, optional
            The database (keyspace) name. Defaults to ``"thingsboard"``.
        set_name : str, optional
            The table name. Defaults to ``"ts_kv_cf"``.
        fetch_size : int, optional
            Number of rows per Cassandra fetch page. Defaults to 50000.
        """
        self.IP = IP
        self.db_name = db_name
        self.set_name = set_name
        self.deviceID = deviceID
        self.fetch_size = fetch_size

        # Persistent connection — reused across all queries
        self._cluster = Cluster([self.IP])
        self._session = self._cluster.connect(self.db_name)
        self._session.default_fetch_size = self.fetch_size

        self.keys = self._keys()

    def close(self):
        """
        Close the Cassandra session and cluster connection.

        Call this when you are done querying to release resources.
        Safe to call multiple times.
        """
        if self._session is not None:
            self._session.shutdown()
            self._session = None
        if self._cluster is not None:
            self._cluster.shutdown()
            self._cluster = None

    def __del__(self):
        """Close the connection on garbage collection."""
        self.close()

    def __enter__(self):
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close on context manager exit."""
        self.close()

    def _keys(self):
        """
        Fetch available metric keys for the device from ts_kv_latest_cf.

        Returns
        -------
        list[str]
            List of metric key names available for this device.
        """
        query = (
            "SELECT key FROM ts_kv_latest_cf "
            "WHERE entity_type='DEVICE' AND entity_id=%s"
        ) % self.deviceID
        rows = self._session.execute(query)
        return [row.key for row in rows]

    def _read_partition(self, start_ts, end_ts):
        """
        Read all data for a single time partition from Cassandra.

        Queries all metric keys in a single CQL query per monthly
        Cassandra partition using ``key IN (...)``.

        Parameters
        ----------
        start_ts : int
            Start timestamp in milliseconds.
        end_ts : int
            End timestamp in milliseconds.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``ts``, ``key``, ``dbl_v``.
            Empty DataFrame if no data found.
        """
        monthly_partitions = self._splitTimesToPartitions(start_ts, end_ts)
        keys_csv = ", ".join(f"'{k}'" for k in self.keys)

        all_rows = []
        for part_start, part_end in monthly_partitions:
            query = (
                f"SELECT ts, key, dbl_v FROM {self.set_name} "
                f"WHERE entity_type='DEVICE' AND entity_id={self.deviceID} "
                f"AND key IN ({keys_csv}) "
                f"AND partition={part_start} "
                f"AND ts>={max(part_start, start_ts)} "
                f"AND ts<={min(part_end, end_ts)}"
            )
            rows = self._session.execute(query)
            all_rows.extend(rows)

        if not all_rows:
            return pandas.DataFrame(columns=['ts', 'key', 'dbl_v'])

        return pandas.DataFrame(
            [(r.ts, r.key, r.dbl_v) for r in all_rows],
            columns=['ts', 'key', 'dbl_v']
        )

    def bag(self, start_time, end_time, npartitions=10):
        """
        Create a Dask DataFrame for parallel reads over a time range.

        Splits the time range into ``npartitions`` equal intervals and
        reads each interval in parallel using ``dask.delayed``.

        Parameters
        ----------
        start_time : str or int
            Start of the time range. Accepts date strings (e.g.,
            ``"2024-01-01"``) or millisecond timestamps.
        end_time : str or int
            End of the time range. Same format as ``start_time``.
        npartitions : int, optional
            Number of parallel partitions. Defaults to 10.

        Returns
        -------
        dask.dataframe.DataFrame
            A Dask DataFrame with columns ``ts``, ``key``, ``dbl_v``.
        """
        if isinstance(start_time, str):
            start_time = int(pandas.Timestamp(start_time).tz_localize("Israel").timestamp() * 1000)
        if isinstance(end_time, str):
            end_time = int(pandas.Timestamp(end_time).tz_localize("Israel").timestamp() * 1000)

        boundaries = np.linspace(start_time, end_time, npartitions + 1, dtype=np.int64)

        delayed_parts = []
        for i in range(npartitions):
            part = dask.delayed(self._read_partition)(int(boundaries[i]), int(boundaries[i + 1]))
            delayed_parts.append(part)

        meta = pandas.DataFrame({'ts': pandas.Series(dtype='int64'),
                                 'key': pandas.Series(dtype='str'),
                                 'dbl_v': pandas.Series(dtype='float64')})

        return dd.from_delayed(delayed_parts, meta=meta)

    def getDataFrame(self, start_time, end_time, npartitions=10):
        """
        Query device data and return as a pivoted Pandas DataFrame.

        Reads all metric keys for the device over the time range in
        parallel, then pivots so rows are timestamps and columns are
        metric keys.

        Parameters
        ----------
        start_time : str or int
            Start of the time range.
        end_time : str or int
            End of the time range.
        npartitions : int, optional
            Number of parallel partitions. Defaults to 10.

        Returns
        -------
        pandas.DataFrame
            A pivoted DataFrame with timestamps as the index, metric
            keys as columns, and ``dbl_v`` (double) as values.
        """
        ddf = self.bag(start_time=start_time, end_time=end_time, npartitions=npartitions)
        df = ddf.compute()
        if df.empty:
            return df
        return df.pivot_table(index='ts', columns='key', values='dbl_v')

    def _splitTimesToPartitions(self, start_ts, end_ts):
        """
        Split a time range into monthly Cassandra partition boundaries.

        Parameters
        ----------
        start_ts : int
            Start timestamp in milliseconds.
        end_ts : int
            End timestamp in milliseconds.

        Returns
        -------
        list[tuple[int, int]]
            List of (partition_start_ms, partition_end_ms) tuples aligned
            to monthly boundaries.
        """
        start_date = pandas.Timestamp.fromtimestamp(start_ts / 1000.0)
        end_date = pandas.Timestamp.fromtimestamp(end_ts / 1000.0)

        # First day of start month
        start_partition = pandas.Timestamp(year=start_date.year, month=start_date.month, day=1)
        # First day of month after end
        if end_date.month == 12:
            end_partition = pandas.Timestamp(year=end_date.year + 1, month=1, day=1)
        else:
            end_partition = pandas.Timestamp(year=end_date.year, month=end_date.month + 1, day=1)

        # Build list of monthly boundaries
        boundaries_ms = []
        current = start_partition
        while current <= end_partition:
            boundaries_ms.append(int(current.timestamp() * 1000))
            if current.month == 12:
                current = pandas.Timestamp(year=current.year + 1, month=1, day=1)
            else:
                current = pandas.Timestamp(year=current.year, month=current.month + 1, day=1)

        return list(zip(boundaries_ms[:-1], boundaries_ms[1:]))
