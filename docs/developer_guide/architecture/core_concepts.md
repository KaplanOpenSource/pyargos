# Core Concepts

This page describes the architecture and design patterns that underpin the entire pyArgos system — how all modules fit together, their dependencies, and the key design decisions.

---

## System-Wide Module Map

![Diagram](../../images/diagrams/developer_guide_architecture_core_concepts_0_cb2aa856.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
graph TD
    subgraph "Experiment Metadata"
        ES[experimentSetup<br/>dataObjects + factories]
        MGR[manager<br/>experimentManager]
    end

    subgraph "Data Ingestion"
        NR[nodered<br/>flow manager + custom nodes]
        KFK[kafka<br/>consumer]
    end

    subgraph "Data Storage & Query"
        PQ[utils.parquetUtils<br/>write/append Parquet]
        CASS[noSQLdask.cassandraBag<br/>Cassandra queries]
        MONGO[noSQLdask.mongoBag<br/>MongoDB queries]
    end

    subgraph "Infrastructure"
        JSON[utils.jsonutils<br/>flexible JSON loading]
        LOG[utils.logging<br/>EXECUTION level + config]
    end

    subgraph "External Systems"
        TB[ThingsBoard]
        KAFKA[Apache Kafka]
        NRED[Node-RED Server]
        CASSDB[(Cassandra)]
        MONGDB[(MongoDB)]
    end

    ES --> JSON
    ES --> LOG
    MGR --> ES
    MGR --> TB
    MGR --> JSON
    KFK --> PQ
    NR --> NRED
    CASS --> CASSDB
    MONGO --> MONGDB
    KFK --> KAFKA
```
-->

---

## Module Dependency Matrix

| Module | Depends on (internal) | Depends on (external) |
|--------|----------------------|----------------------|
| `experimentSetup` | `utils.jsonutils`, `utils.logging` | `matplotlib`, `requests` |
| `manager` | `experimentSetup`, `utils.jsonutils`, `utils.logging` | `tb_rest_client` (optional) |
| `kafka.consumer` | `utils.parquetUtils` | `kafka-python`, `numpy`, `pandas` |
| `CLI` | `experimentSetup`, `kafka.consumer`, `manager` | `kafka-python` |
| `nodered.manager` | — | `requests` |
| `nodered.nodes` | — | `dask`, `fastparquet`, `pandas` |
| `noSQLdask.cassandraBag` | — | `cassandra-driver`, `dask`, `pandas` |
| `noSQLdask.mongoBag` | — | `pymongo`, `dask`, `pandas` |
| `utils.jsonutils` | — | `pandas` |
| `utils.parquetUtils` | — | `dask`, `pandas`, `shutil` |
| `utils.logging` | — | `logging` (stdlib) |

**Design principle:** Modules at the bottom of the stack (`utils`, `noSQLdask`) have **no internal dependencies** — they only depend on external libraries. Higher-level modules (`manager`, `CLI`) depend on lower-level ones. This avoids circular imports.

---

## Class Hierarchy

pyArgos organizes experiment data in a hierarchical object model:

![Diagram](../../images/diagrams/developer_guide_architecture_core_concepts_1_d46b2a03.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
classDiagram
    class Experiment {
        +name
        +description
        +trialSet: dict
        +entityType: dict
        +imageMap: dict
        +entitiesTable: DataFrame
        +entityTypeTable: DataFrame
        +refresh()
        +toJSON()
    }

    class ExperimentZipFile {
        +refresh()
        +getImage()
    }

    class TrialSet {
        +name
        +experiment: Experiment
        +properties: dict
        +trialsTable: DataFrame
        +toJSON()
    }

    class Trial {
        +name
        +trialSet: TrialSet
        +properties: dict
        +entities: dict
        +entitiesTable()
        +propertiesTable: DataFrame
    }

    class EntityType {
        +name
        +experiment: Experiment
        +properties: dict
        +entitiesTable: DataFrame
        +toJSON()
    }

    class Entity {
        +name
        +entityType: str
        +properties: dict
        +allProperties: dict
        +propertiesTable: DataFrame
        +trialProperties()
        +toJSON()
    }

    Experiment <|-- ExperimentZipFile
    Experiment "1" --> "*" TrialSet
    Experiment "1" --> "*" EntityType
    TrialSet "1" --> "*" Trial
    EntityType "1" --> "*" Entity
    Trial --> Entity : references
```
-->

See [Experiment Setup Architecture](experiment_setup.md) for the full deep-dive.

---

## Factory Pattern

pyArgos uses the Factory pattern to abstract experiment loading from different sources:

![Diagram](../../images/diagrams/developer_guide_architecture_core_concepts_2_187433b0.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
graph TD
    A[Client Code] --> C[fileExperimentFactory]
    C --> E[Experiment / ExperimentZipFile]
    E --> G[Unified Experiment Interface]
```
-->

### `fileExperimentFactory`

- Loads from a local directory
- Auto-detects ZIP vs extracted JSON format
- Handles version migrations (1.0.0 through 3.0.0)
- Default: uses current working directory

---

## Swimlane: Full Experiment Deployment

This diagram shows the complete flow from loading an experiment to deploying it on ThingsBoard and starting data collection:

![Diagram](../../images/diagrams/developer_guide_architecture_core_concepts_3_abe78673.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI / Script
    participant Factory as fileExperimentFactory
    participant Exp as Experiment
    participant Mgr as experimentManager
    participant TB as ThingsBoard
    participant Kafka as Kafka
    participant Consumer as consume_topic

    Note over User,Consumer: Phase 1: Load & Deploy
    User->>CLI: Load experiment
    CLI->>Factory: fileExperimentFactory(path)
    Factory->>Exp: getExperiment()
    Exp-->>CLI: Experiment object

    User->>CLI: Deploy to ThingsBoard
    CLI->>Mgr: experimentManager(path)
    Mgr->>TB: Create device profiles
    Mgr->>TB: Create devices
    Mgr->>TB: Upload trial attributes

    Note over User,Consumer: Phase 2: Create Kafka Topics
    User->>CLI: kafka_createTopics
    CLI->>Exp: Get entity types
    CLI->>Kafka: Create topic per device type

    Note over User,Consumer: Phase 3: Consume Data
    User->>CLI: kafka_runConsumers
    loop For each device type (parallel threads)
        CLI->>Consumer: consume_topic(type, data/)
        Consumer->>Kafka: Poll messages
        Consumer->>Consumer: Write to Parquet
    end
```
-->

---

## Swimlane: CLI Command Routing

The CLI (`argos.CLI`) routes user commands to the appropriate module:

![Diagram](../../images/diagrams/developer_guide_architecture_core_concepts_4_c15883df.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    actor User
    participant CLI as argos.CLI
    participant ES as experimentSetup
    participant KFK as kafka.consumer
    participant MGR as manager
    participant NR as nodered (CLI)

    User->>CLI: --createExperiment
    CLI->>ES: experiment_createDirectory()

    User->>CLI: --kafkaCreateTopics
    CLI->>ES: fileExperimentFactory → get types
    CLI->>KFK: Create Kafka topics via KafkaAdminClient

    User->>CLI: --kafkaRunConsumers
    CLI->>KFK: consume_topic() per type (threaded)

    User->>CLI: --kafkaRunConsumersServer
    CLI->>KFK: consume_topic_server() per type (threaded)

    User->>CLI: --tbLoadTrial
    CLI->>MGR: experimentManager → loadTrialDesignToThingsboard()

    User->>CLI: --tbSetupExperiment
    CLI->>ES: fileExperimentFactory → setupExperiment()

    User->>CLI: --tbCleanDevices
    CLI->>MGR: experimentManager → clearDevicesFromThingsboard()

    User->>CLI: --noderedCreateDeviceMap
    CLI->>NR: nodered_createDeviceMap()
```
-->

---

## Property Type System

Entities and trials have typed properties. The `Trial` class includes parsers for each type:

| Property Type | Parser | Output |
|---------------|--------|--------|
| `location` | `_parseProperty_location` | `{name, latitude, longitude}` |
| `text` | `_parseProperty_text` | String value |
| `textArea` | `_parseProperty_textArea` | String value |
| `number` | `_parseProperty_number` | Numeric value |
| `boolean` | `_parseProperty_boolean` | Boolean value |
| `datetime_local` | `_parseProperty_datetime_local` | Datetime (Israel timezone) |
| `selectList` | `_parseProperty_selectList` | Selected value from predefined options |

---

## Container Hierarchy Resolution

The `fillContained` module resolves entity hierarchies where entities can be contained within other entities:

```python
fill_properties_by_contained(entities_types_dict, meta_entities)
```

This function:

1. Traverses the "contains" relationships between entities
2. Inherits parent properties to child entities
3. Handles property type conversion (Number, String)
4. Spreads flattened attributes (e.g., `location` -> `mapName`, `latitude`, `longitude`)

---

## Pandas as Data Interface

A key design decision in pyArgos is exposing data as Pandas DataFrames wherever possible:

- `experiment.entitiesTable` - All entities as a flat DataFrame
- `experiment.entityTypeTable` - Entity types summary
- `trialSet.trialsTable` - All trials in a set
- `trial.propertiesTable` - Trial-level properties
- `entity.propertiesTable` - Entity constant properties
- `entity.allPropertiesTable` - All properties including trial-specific

This makes it easy to filter, join, and analyze experiment metadata using standard Pandas operations.

---

## Design Decisions

### Why optional external dependencies?

ThingsBoard (`tb_rest_client`), Cassandra (`cassandra-driver`), and MongoDB (`pymongo`) are all optional. pyArgos gracefully handles their absence with try/except imports. This allows users to install only what they need — a researcher doing offline analysis doesn't need ThingsBoard, and a deployment engineer doesn't need Cassandra.

### Why threads for Kafka consumers?

The CLI starts one thread per device type topic (`threading.Thread`). This is simpler than multiprocessing and sufficient because the bottleneck is Kafka I/O and Parquet writes, not CPU. Each thread is independent — one topic's consumer has no effect on another.

### Why Dask for Parquet and NoSQL?

Dask provides lazy evaluation and partitioned I/O, which is important for:
- **Parquet:** Auto-repartitioning when files grow large (>100MB partitions)
- **Cassandra:** Parallel reads across monthly partitions
- **MongoDB:** Parallel reads across time-range partitions

### Why a custom EXECUTION log level?

pyArgos defines `EXECUTION = 15` (between DEBUG=10 and INFO=20) for step-by-step progress messages. This separates operational progress ("Creating experiment directories") from informational messages ("Found 3 devices") without polluting the DEBUG stream with low-level details.

---

## Version Compatibility

The `ExperimentZipFile` class handles multiple JSON schema versions:

| Version | Handler | Changes |
|---------|---------|---------|
| 1.0.0 | `_fix_json_version_1_0_0_` | Original format |
| 2.0.0 | `_fix_json_version_2_0_0_` | Updated entity structure |
| 3.0.0 | `_fix_json_version_3_0_0_` | Current format |

Version detection and migration is automatic when loading experiments.
