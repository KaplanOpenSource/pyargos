# API Reference Overview

This section documents the public API of each pyArgos module.

---

## Module Map

```
argos/
  __init__.py                 # Exports: WEB, FILE, experimentManager
  CLI.py                      # CLI command handlers
  manager.py                  # experimentManager class
  experimentSetup/
    __init__.py               # FILE constant
    dataObjectsFactory.py     # fileExperimentFactory
    dataObjects.py            # Experiment, TrialSet, Trial, EntityType, Entity
    fillContained.py          # fill_properties_by_contained()
  kafka/
    consumer.py               # consume_topic(), consume_topic_server()
  nodered/
    manager/
      flowManagerHome.py      # flowManagerHome class
    nodes/
      heraNodes.py            # to_parquet Node-RED node
      tbNodes.py              # add_document Node-RED node
  noSQLdask/
    cassandraBag.py           # CassandraBag class
    mongoBag.py               # MongoBag class
  utils/
    jsonutils.py              # loadJSON(), processJSONToPandas(), convertJSONtoPandas()
    parquetUtils.py           # writeToParquet(), appendToParquet()
    logging/
      helpers.py              # initialize_logging(), get_logger(), EXECUTION level
      toolkit.py              # Legacy logging toolkit (deprecated)
```

---

## Quick Import Guide

```python
# Load an experiment from files
from argos.experimentSetup import fileExperimentFactory
experiment = fileExperimentFactory("/path").getExperiment()

# Use the experiment manager (with ThingsBoard)
from argos.manager import experimentManager
manager = experimentManager("/path")

# Consume Kafka data
from argos.kafka.consumer import consume_topic

# JSON utilities
from argos.utils.jsonutils import loadJSON, convertJSONtoPandas

# Parquet utilities
from argos.utils.parquetUtils import writeToParquet, appendToParquet

# NoSQL interfaces
from argos.noSQLdask.cassandraBag import CassandraBag
from argos.noSQLdask.mongoBag import MongoBag

# Logging
from argos.utils.logging.helpers import get_logger, initialize_logging
```

---

## API Pages

| Module | Page |
|--------|------|
| Experiment Setup | [Factories, data objects, entities, trials](experiment_setup.md) |
| Experiment Manager | [ThingsBoard interface and orchestration](manager.md) |
| Kafka Consumer | [Topic consumption and Parquet writing](kafka.md) |
| Node-RED | [Custom nodes and flow management](nodered.md) |
| NoSQL | [Cassandra and MongoDB via Dask](nosql.md) |
| Utilities | [JSON, Parquet, and logging helpers](utils.md) |
