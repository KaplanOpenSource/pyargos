"""
Dask-based interface to MongoDB time-series collections.

Provides the ``MongoBag`` class for querying time-series data from
MongoDB using Dask for parallel, partitioned reads.

Performance notes:

- Uses a single shared MongoClient (connection pooling) instead of
  opening/closing a connection per partition.
- Uses datetime objects in queries instead of string comparison, allowing
  MongoDB to use time-based indexes efficiently.
- Uses ``dask.delayed`` → ``pd.DataFrame`` instead of ``dask.bag`` to
  avoid unnecessary materialization overhead.
- Supports projections to limit returned fields.
"""

import dask
import dask.dataframe as dd
import pymongo
import pandas


class MongoBag:
    """
    Dask interface for querying MongoDB time-series collections.

    Partitions a time range into intervals and reads each interval in
    parallel using ``dask.delayed``.

    Uses a single shared ``MongoClient`` for all queries (connection pooled
    by pymongo). Call ``close()`` when done.

    Parameters
    ----------
    db_name : str
        The MongoDB database name.
    collection_name : str
        The collection name within the database.
    datetimeField : str, optional
        The name of the timestamp field in the documents.
        Defaults to ``"timestamp"``.
    host : str, optional
        The MongoDB connection URI. Defaults to ``"localhost"``.

    Examples
    --------
    >>> bag = MongoBag(db_name="mydb", collection_name="sensor_data")
    >>> df = bag.getDataFrame("2024-01-01", "2024-01-31", periods=20)
    >>> bag.close()

    Or as a context manager:

    >>> with MongoBag("mydb", "sensor_data") as bag:
    ...     df = bag.getDataFrame("2024-01-01", "2024-01-31")
    """

    def __init__(self, db_name, collection_name, datetimeField="timestamp",
                 host="localhost"):
        """
        Initialize the MongoBag with a persistent MongoClient.

        Parameters
        ----------
        db_name : str
            The MongoDB database name.
        collection_name : str
            The collection name.
        datetimeField : str, optional
            The timestamp field name. Defaults to ``"timestamp"``.
        host : str, optional
            The MongoDB connection URI. Defaults to ``"localhost"``.
        """
        self._db_name = db_name
        self._collection_name = collection_name
        self._timestamp_field = datetimeField
        self._client = pymongo.MongoClient(host)

    def close(self):
        """
        Close the MongoDB client connection.

        Safe to call multiple times.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self):
        """Close the connection on garbage collection."""
        self.close()

    def __enter__(self):
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close on context manager exit."""
        self.close()

    @property
    def db_name(self):
        """
        The MongoDB database name.

        Returns
        -------
        str
            The database name.
        """
        return self._db_name

    @property
    def collection_name(self):
        """
        The MongoDB collection name.

        Returns
        -------
        str
            The collection name.
        """
        return self._collection_name

    @property
    def timestamp_field(self):
        """
        The name of the timestamp field used for time-range queries.

        Returns
        -------
        str
            The timestamp field name.
        """
        return self._timestamp_field

    def _read_partition(self, start_ts, end_ts, projection=None, **qry):
        """
        Read documents from MongoDB for a single time interval.

        Uses datetime objects for the query (not strings) so MongoDB
        can use time-based indexes.

        Parameters
        ----------
        start_ts : datetime-like
            Start of the time interval.
        end_ts : datetime-like
            End of the time interval.
        projection : dict, optional
            MongoDB projection to limit returned fields.
            Example: ``{"_id": 0, "timestamp": 1, "value": 1}``.
        **qry : dict
            Additional MongoDB query filters.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of documents matching the query.
            Empty DataFrame if no results.
        """
        full_qry = {
            self._timestamp_field: {
                '$gte': start_ts,
                '$lt': end_ts
            }
        }
        full_qry.update(qry)

        collection = self._client[self._db_name][self._collection_name]

        cursor = collection.find(full_qry, projection=projection)
        items = list(cursor)

        if not items:
            return pandas.DataFrame()

        return pandas.DataFrame(items)

    def bag(self, start_time, end_time, periods=10, freq=None,
            projection=None, **qry):
        """
        Create a Dask DataFrame for parallel reads over a time range.

        Splits the time range into partitions using ``pandas.date_range``
        and reads each partition in parallel using ``dask.delayed``.

        Parameters
        ----------
        start_time : str
            Start of the time range (parsed by ``pandas.to_datetime``).
        end_time : str
            End of the time range.
        periods : int, optional
            Number of partitions. Defaults to 10. Ignored if ``freq`` is set.
        freq : str, optional
            Partition frequency (e.g., ``"1D"``, ``"1H"``). If set,
            ``periods`` is ignored.
        projection : dict, optional
            MongoDB projection to limit returned fields.
        **qry : dict
            Additional MongoDB query filters.

        Returns
        -------
        dask.dataframe.DataFrame
            A Dask DataFrame of documents from the collection.
        """
        start_time = pandas.to_datetime(start_time)
        end_time = pandas.to_datetime(end_time)

        date_range = pandas.date_range(
            start_time, end_time, periods=periods, freq=freq, tz="israel"
        )

        delayed_parts = []
        for i in range(len(date_range) - 1):
            part = dask.delayed(self._read_partition)(
                date_range[i].to_pydatetime(),
                date_range[i + 1].to_pydatetime(),
                projection=projection,
                **qry
            )
            delayed_parts.append(part)

        return dd.from_delayed(delayed_parts)

    def getDataFrame(self, start_time, end_time, periods=10, freq=None,
                     projection=None, **qry):
        """
        Query data and return as a Pandas DataFrame.

        Convenience method that calls ``bag()`` and computes immediately.

        Parameters
        ----------
        start_time : str
            Start of the time range.
        end_time : str
            End of the time range.
        periods : int, optional
            Number of partitions. Defaults to 10.
        freq : str, optional
            Partition frequency.
        projection : dict, optional
            MongoDB projection to limit returned fields.
        **qry : dict
            Additional MongoDB query filters.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of all matching documents.
        """
        ddf = self.bag(start_time, end_time, periods=periods, freq=freq,
                       projection=projection, **qry)
        return ddf.compute()

    def read_datetime_interval_from_collection(self, args, **qry):
        """
        Read documents from MongoDB for a single time interval.

        .. deprecated::
            Use ``_read_partition()`` instead. This method is kept for
            backwards compatibility.

        Parameters
        ----------
        args : tuple[Timestamp, Timestamp]
            A ``(start, end)`` tuple of partition boundaries.
        **qry : dict
            Additional MongoDB query filters.

        Returns
        -------
        list[dict]
            A list of document dicts matching the time range and filters.
        """
        return self._read_partition(args[0], args[1], **qry).to_dict('records')
