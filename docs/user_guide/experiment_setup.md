# Experiment Setup

This guide walks through creating, configuring, and running a pyArgos experiment.

---

## Creating an Experiment Directory

Use the CLI to scaffold a new experiment:

```bash
python -m argos.bin.trialManager --createExperiment \
    --experimentName MyExperiment \
    --directory /path/to/experiments
```

This creates:

```
MyExperiment/
  code/
    argos_basic.py          # Starter script to load the experiment
  data/                     # Where Parquet files will be stored
  runtimeExperimentData/
    Datasources_Configurations.json   # Main configuration
```

---

## Configuring the Experiment

### Datasources Configuration

Edit `runtimeExperimentData/Datasources_Configurations.json`:

```json
{
    "experimentName": "MyExperiment",
    "kafka": {
        "bootstrap_servers": ["127.0.0.1:9092"]
    },
    "Thingsboard": {
        "restURL": "http://localhost:8080",
        "username": "tenant@thingsboard.org",
        "password": "tenant"
    }
}
```

### Defining Entities

In `experimentData/ExperimentData.json`, define devices and assets:

```json
{
    "Entities": [
        {
            "entityType": "DEVICE",
            "Number": 5,
            "Type": "Sensor",
            "nameFormat": "Sensor_{id:02d}"
        },
        {
            "entityType": "ASSET",
            "Number": 1,
            "Type": "Gateway",
            "nameFormat": "Gateway_{id:02d}"
        }
    ]
}
```

| Field | Description |
|-------|-------------|
| `entityType` | `"DEVICE"` or `"ASSET"` |
| `Number` | How many entities to create |
| `Type` | The device/asset type name |
| `nameFormat` | Name template. `{id}` is replaced by running number (starting at 1) |

!!! example
    `Number=3` and `nameFormat="Sensor_{id:02d}"` creates: `Sensor_01`, `Sensor_02`, `Sensor_03`

### Calculation Windows (optional)

The `"properties"/"calculationWindows"` section creates shadow devices for streaming calculations:

```json
{
    "properties": {
        "calculationWindows": [60, 180, 300]
    }
}
```

This creates additional devices like `Sensor_01_60s`, `Sensor_01_180s`, `Sensor_01_300s` for each entity.

---

## Setting Up Trials

### 1. Generate the trial template

```bash
cd MyExperiment
python /path/to/pyargos/bin/trialManager.py \
    --expConf experimentConfiguration.json --setup
```

This creates `experimentData/trials/trialTemplate.json`.

### 2. Create a trial design

Copy the template to the design directory:

```bash
cp experimentData/trials/trialTemplate.json \
   experimentData/trials/design/morningTrial.json
```

### 3. Edit the trial

Edit the trial file to set per-entity attributes:

```json
{
    "Sensor_01": {
        "attributes": {
            "location": {"lat": 32.0, "lon": 34.8},
            "threshold": 25.0
        }
    },
    "Sensor_02": {
        "attributes": {
            "location": {"lat": 32.1, "lon": 34.9},
            "threshold": 30.0
        }
    }
}
```

You can configure for each entity:

- `Name` / `Type` / `entityType` - Override entity metadata
- `attributes` - Key-value pairs for device configuration
- `contains` - Relations: `[["DEVICE", "Sensor_01"], ["ASSET", "Gateway_01"]]`

### 4. Upload to ThingsBoard

```bash
python /path/to/pyargos/bin/trialManager.py \
    --expConf experimentConfiguration.json \
    --load morningTrial
```

---

## Loading Experiments in Python

### From local files

```python
from argos.experimentSetup import fileExperimentFactory

experiment = fileExperimentFactory("/path/to/MyExperiment").getExperiment()

# Access entities
print(experiment.entitiesTable)

# Access a trial
trial = experiment.trialSet["design"]["morningTrial"]
print(trial.entitiesTable())
```

---

## Using the Generated Starter Script

The auto-generated `code/argos_basic.py` provides a quick way to start working:

```python
# In code/argos_basic.py (auto-generated)
from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory

MyExperiment = fileExperimentFactory("/path/to/MyExperiment").getExperiment()
print("Experiment loaded into variable MyExperiment")
```

Run it interactively:

```bash
python -i code/argos_basic.py
```
