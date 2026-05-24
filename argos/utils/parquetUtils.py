import shutil

import dask
import dask.dataframe as dd
import numpy
import logging
from datetime import datetime
import pandas


def appendToParquet(toBeAppended,additionalData,datetimeColumn='datetime'):
    """
        Append to the parquet file and write it back to the original location.
        Keep the partition size.

        We assume it is the same structure.

        We use the datetime column to create a [day]_[month]_[year] string for the
        partitions.

    Parameters
    ----------
    toBeAppended : str
        The file that is going to be appended.

    additionalData : dask/pandas dataframe, list of dataframes.
        The new data

    datetimeColumn : str
        The name of the column with the date time.
        If None, use index.

    Returns
    -------
        bool
        True if successful
    """
    logger = logging.getLogger("argos.utils.parquetUtils.appendToParquet")
    logger.execution("----------- Start -------------")
    logger.debug(f"Appending parquet file {toBeAppended}.")

    if isinstance(additionalData,pandas.DataFrame):
        newData = additionalData
    elif isinstance(additionalData,dask.dataframe.DataFrame):
        newData = additionalData.compute()
    else:
        newData = pandas.DataFrame(additionalData)

    if datetimeColumn not in newData:
        newData = newData.reset_index()



    logger.debug(f"Loading old data file {toBeAppended}.")
    dsk = dd.read_parquet(toBeAppended, engine="pyarrow")
    united = dd.concat([dsk, newData.set_index(datetimeColumn)])
    logger.debug(f"Check if the last partition is too large. ")
    if united.partitions[-1].memory_usage().sum().compute()/1e6 > 100 or united.npartitions > 10:
        logger.debug(f"Repartition file")
        united = united.repartition(partition_size="100MB")

    logger.debug(f"Saving temporary file")
    united.to_parquet(f"{toBeAppended}.tmp")

    logger.debug(f"Removing old file")
    shutil.rmtree(toBeAppended)
    logger.debug(f"Changing name of the new file. ")
    shutil.move(f"{toBeAppended}.tmp",toBeAppended)

    #newData = newData.assign(datetimeString=newData[datetimeColumn].apply(lambda x: x.strftime("%d_%m_%Y"))).set_index(datetimeColumn)
    #dsk = dd.from_pandas(newData, npartitions=1).repartition(freq="1D")
    #dsk.to_parquet(toBeAppended, append=True, partition_on="datetimeString",ignore_divisions=True)
    return True

def writeToParquet(parquetFile,data,datetimeColumn='datetime'):
    """
        Writes a new parquet for the argos file.

        We use the datetime column to create a [day]_[month]_[year] string for the
        partition.


    Parameters
    ----------
    parquetFile  : str
            The parquet files

    data : pandas/dask data
            The new data

    datetimeColumn : str
        The name of the datetime column for the partition.
        Provide the name even if it is the index.

    Returns
    -------
        None
    """
    logger = logging.getLogger("argos.utils.parquetUtils.appendToParquet")
    logger.execution("----------- Start -------------")
    logger.debug(f"Writing to  parquet file {parquetFile}.")

    if isinstance(data,pandas.DataFrame):
        newData = data
    elif isinstance(data,dask.dataframe.DataFrame):
        newData = data.compute()
    else:
        newData = pandas.DataFrame(data)

    if datetimeColumn not in newData:
        newData = newData.reset_index()

    #newData = newData.assign(datetimeString=newData[datetimeColumn].apply(lambda x: x.strftime("%d_%m_%Y"))).set_index(datetimeColumn)
    dsk = dd.from_pandas(newData,npartitions=1).set_index(datetimeColumn)#.repartition(freq="1D")
    dsk.to_parquet(parquetFile, engine="pyarrow")
    return True
