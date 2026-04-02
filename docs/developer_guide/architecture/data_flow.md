# Data Flow

This page documents how data flows through the pyArgos system in different scenarios.

---

## Local Experiment Loading

![Diagram](../../images/diagrams/developer_guide_architecture_data_flow_0_f3f63982.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    participant Client
    participant Factory as fileExperimentFactory
    participant FS as File System
    participant Exp as Experiment

    Client->>Factory: fileExperimentFactory(path)
    Client->>Factory: getExperiment()
    Factory->>FS: Scan runtimeExperimentData/
    alt ZIP file found
        Factory->>FS: Read .zip archive
        Factory->>Exp: Create ExperimentZipFile
        Exp->>Exp: Parse data.json from ZIP
        Exp->>Exp: Apply version fixes
    else JSON files found
        Factory->>FS: Read JSON files
        Factory->>Exp: Create Experiment
    end
    Exp->>Exp: Build TrialSet hierarchy
    Exp->>Exp: Build EntityType hierarchy
    Factory-->>Client: Return Experiment
```
-->

---

## Kafka to Parquet Pipeline

![Diagram](../../images/diagrams/developer_guide_architecture_data_flow_2_bfa05a42.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    participant Device as IoT Device
    participant NR as Node-RED
    participant Kafka
    participant Consumer as pyArgos Consumer
    participant PQ as Parquet File

    Device->>NR: MQTT/UDP data
    NR->>Kafka: Route to device type topic
    Consumer->>Kafka: poll(timeout=12s, max=5000)
    Kafka-->>Consumer: Message batch
    Consumer->>Consumer: JSON to DataFrame
    Consumer->>Consumer: Add timezone (Israel)
    Consumer->>Consumer: Cast numeric types
    Consumer->>Consumer: Remove duplicate timestamps
    alt File exists
        Consumer->>PQ: appendToParquet()
    else New file
        Consumer->>PQ: writeToParquet()
    end
```
-->

### Parquet Write Strategy

The `appendToParquet` function uses a safe write pattern:

1. Load existing Parquet file with Dask
2. Concatenate with new data
3. Auto-repartition if:
    - Last partition exceeds 100MB
    - Total partitions exceed 10
4. Write to temporary file
5. Atomic rename to final path

---

## ThingsBoard Setup Flow

![Diagram](../../images/diagrams/developer_guide_architecture_data_flow_3_8036da00.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    participant Manager as experimentManager
    participant TB as ThingsBoard
    participant Exp as Experiment

    Manager->>Exp: Load experiment from files
    Manager->>TB: Login (REST client)

    Note over Manager,TB: Device Setup
    Manager->>TB: Get existing device profiles
    loop Each entity type
        alt Profile missing
            Manager->>TB: Create device profile
        end
        loop Each entity
            alt Device missing
                Manager->>TB: Create device
            end
        end
    end

    Note over Manager,TB: Trial Loading
    loop Each entity in trial
        Manager->>TB: Clear attributes (all scopes)
        Manager->>TB: Save SERVER_SCOPE attributes
    end
```
-->

---

## NoSQL Query Flow

### Cassandra

![Diagram](../../images/diagrams/developer_guide_architecture_data_flow_4_265874ee.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
graph TD
    A[CassandraBag] --> B[Query device keys]
    B --> C[Split time range into partitions]
    C --> D[Create Dask bag]
    D --> E[Parallel reads per partition]
    E --> F[Pivot: rows=timestamps, cols=keys]
    F --> G[Return DataFrame]
```
-->

### MongoDB

![Diagram](../../images/diagrams/developer_guide_architecture_data_flow_5_ae1b1c36.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
graph TD
    A[MongoBag] --> B[Create date range partitions]
    B --> C[Create Dask bag]
    C --> D[Parallel reads per partition]
    D --> E[Filter by MongoDB query]
    E --> F[Return DataFrame]
```
-->

Both use Dask for distributed processing, splitting time ranges into partitions for parallel reads from the database.
