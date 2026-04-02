# Experiment Setup API

**Module:** `argos.experimentSetup`

The experiment setup module is the core of pyArgos. It provides the factory pattern for loading experiments from different sources and the data object hierarchy that represents experiments, trials, and entities.

---

## Class Roles at a Glance

Each class in this module has a distinct responsibility in the experiment lifecycle:

| Class | Role | Analogy |
|-------|------|---------|
| `fileExperimentFactory` | Locates and loads experiment data from local disk | File browser |
| `Experiment` | Root container holding all entity types, trial sets, and image maps | Project folder |
| `ExperimentZipFile` | Variant of Experiment that reads from a `.zip` archive (with version migration) | ZIP reader |
| `TrialSet` | Named collection of trials (e.g., "design", "deploy") | Folder of trials |
| `Trial` | A single experimental configuration with per-entity attributes | Configuration snapshot |
| `EntityType` | Schema definition + collection of entities of one type (e.g., "Sensor") | Device class |
| `Entity` | A single device/asset instance with constant and trial-specific properties | Individual device |

---

## Inheritance Diagram

### Experiment Hierarchy

`Experiment` is the base class. `ExperimentZipFile` overrides the data-loading and image-fetching behavior, keeping the rest of the interface identical:

![Diagram](../../images/diagrams/developer_guide_api_experiment_setup_0_a46b01b4.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
classDiagram
    class Experiment {
        +name : str
        +description : str
        +trialSet : dict~str, TrialSet~
        +entityType : dict~str, EntityType~
        +entitiesTable : DataFrame
        +entityTypeTable : DataFrame
        +imageMap : dict
        +refresh()
        +getImage(imageName) ndarray
        +toJSON() dict
    }

    class ExperimentZipFile {
        +refresh()  ⟵ overrides
        +getImage() ⟵ overrides
        -_fix_json_version_1_0_0_()
        -_fix_json_version_2_0_0_()
        -_fix_json_version_3_0_0()
    }

    Experiment <|-- ExperimentZipFile : "reads from ZIP archive"
```
-->

**What each subclass changes:**

- **ExperimentZipFile** -- `refresh()` extracts `data.json` from the ZIP, detects the schema version, and applies the appropriate migration. `getImage()` reads images from inside the archive. `_init_ImageMaps()` parses the ZIP's internal map format.

### Container Hierarchy (dict-based)

`TrialSet` and `EntityType` extend Python's built-in `dict`, so they work as both typed objects with properties **and** dictionaries for accessing children by name:

![Diagram](../../images/diagrams/developer_guide_api_experiment_setup_1_e83be745.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
classDiagram
    class dict {
        <<builtin>>
    }

    class TrialSet {
        +name : str
        +experiment : Experiment
        +properties : list
        +propertiesTable : DataFrame
        +trialsTable : DataFrame
        +toJSON() dict
    }

    class EntityType {
        +name : str
        +experiment : Experiment
        +properties : list
        +propertiesTable : DataFrame
        +entitiesTable : DataFrame
        +entitiesAllProperties : DataFrame
        +toJSON() dict
    }

    class Trial {
        +name : str
        +trialSet : TrialSet
        +properties : dict
        +propertiesTable : DataFrame
        +entities : dict
        +entitiesTable : DataFrame
    }

    class Entity {
        +name : str
        +entityType : str
        +properties : dict
        +propertiesTable : DataFrame
        +allPropertiesTable : DataFrame
        +trialProperties(setName, trialName) dict
    }

    dict <|-- TrialSet
    dict <|-- EntityType
    TrialSet "1" *-- "*" Trial : contains
    EntityType "1" *-- "*" Entity : contains
    Trial ..> Entity : references via entities
```
-->

This means you can write:

```python
# Dict-style access to children
trial = experiment.trialSet["design"]["morning"]
entity = experiment.entityType["Sensor"]["Sensor_01"]

# But also use rich properties
experiment.trialSet["design"].trialsTable      # DataFrame of all trials
experiment.entityType["Sensor"].entitiesTable  # DataFrame of all sensors
```

---

## Call Workflow: Loading an Experiment from File

This swimlane diagram shows the complete call chain when a user loads an experiment from local files, from the initial factory call through object construction and the resulting object graph:

![Diagram](../../images/diagrams/developer_guide_api_experiment_setup_2_e0a76e17.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    actor User
    participant Factory as fileExperimentFactory
    participant FS as File System
    participant Exp as Experiment / ExperimentZipFile
    participant TS as TrialSet
    participant TR as Trial
    participant ET as EntityType
    participant EN as Entity

    User->>Factory: factory = fileExperimentFactory(path)
    User->>Factory: experiment = factory.getExperiment()
    Factory->>FS: Scan runtimeExperimentData/ for *.zip
    alt ZIP file found
        FS-->>Factory: zipfile path
        Factory->>Exp: ExperimentZipFile(zipfile)
    else No ZIP, look for experiment.json
        FS-->>Factory: JSON file
        Factory->>Exp: Experiment(jsonFile)
    end

    Note over Exp: __init__ begins
    Exp->>Exp: refresh()

    alt ExperimentZipFile
        Exp->>FS: Extract data.json from ZIP
        Exp->>Exp: Detect version (1.0/2.0/3.0)
        Exp->>Exp: Apply version migration
    else Experiment (plain JSON)
        Exp->>FS: loadJSON(file)
    end

    Exp->>Exp: _initTrialSets()
    loop For each trialSet in config
        Exp->>TS: TrialSet(experiment, metadata)
        TS->>TS: _initTrials()
        loop For each trial in trialSet
            TS->>TR: Trial(trialSet, metadata)
            TR->>TR: Parse properties by type
            Note over TR: location → lat/lon/name<br/>number → float<br/>boolean → True/False<br/>datetime → Timestamp(Israel)
        end
    end

    Exp->>Exp: _initEntitiesTypes()
    loop For each entityType in config
        Exp->>ET: EntityType(experiment, metadata)
        ET->>ET: _initEntities()
        loop For each entity in type
            ET->>EN: Entity(entityType, metadata)
            EN->>EN: Collect Constant-scope props (type defaults)
            EN->>EN: Collect Device-scope props (entity attrs)
        end
    end

    Exp->>Exp: _init_ImageMaps()
    Exp-->>Factory: return Experiment
    Factory-->>User: experiment object ready

    Note over User: Access the data
    User->>Exp: experiment.entitiesTable
    User->>Exp: experiment.trialSet["design"]["trial1"].entities
    User->>EN: entity.trialProperties("design", "trial1")
```
-->

---

## Call Workflow: Accessing Trial Entity Data

When you access `trial.entities` or `trial.entitiesTable`, the containment hierarchy is resolved on-the-fly:

![Diagram](../../images/diagrams/developer_guide_api_experiment_setup_4_20df2591.svg)

<!-- mermaid source (for editing, paste into mermaid.live):
```mermaid
sequenceDiagram
    actor User
    participant TR as Trial
    participant FC as fillContained
    participant CP as _composeProperties
    participant PP as _parseProperty_*

    User->>TR: trial.entitiesTable

    TR->>FC: fill_properties_by_contained(entityTypes, raw_entities)
    Note over FC: Deep copy raw entities

    loop For each entity
        FC->>FC: Convert attribute types (Number→float, String→str)
        loop Walk containment chain (child → parent → grandparent)
            FC->>FC: Inherit parent's location
            FC->>FC: Copy missing attributes from parent
        end
        FC->>FC: spread_attributes(entity)
        Note over FC: location → mapName, latitude, longitude<br/>attributes[] → flat top-level keys<br/>containedIn → containedInType + containedIn
    end

    FC-->>TR: Flattened entity list

    TR->>CP: _composeProperties(entities_df)

    loop For each entity row
        CP->>PP: _composeEntityProperties(entityType, properties)
        Note over PP: Parse each property by type<br/>(location, text, number, boolean, datetime, selectList)
        PP-->>CP: Single-row DataFrame of parsed values
        CP->>CP: Join with entity.propertiesTable (constant props)
    end

    CP-->>TR: Combined DataFrame
    TR-->>User: DataFrame with all entity properties
```
-->

---

## Factories

### fileExperimentFactory

**Role:** Scans the experiment directory for data files (ZIP or JSON) and returns the appropriate Experiment subclass. This is the primary entry point for loading experiments from local disk.

**Key behavior:**

- Looks in `<path>/runtimeExperimentData/` for `.zip` files first
- Falls back to `experiment.json` if no ZIP is found
- Returns `ExperimentZipFile` for ZIP sources, `Experiment` for JSON

::: argos.experimentSetup.dataObjectsFactory.fileExperimentFactory
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - getExperiment

---

## Data Objects

### Experiment

**Role:** The root container for all experiment data. Holds the entity type hierarchy (types → entities) and the trial set hierarchy (sets → trials). Exposes all data as Pandas DataFrames and provides image map access for spatial visualization.

**When to use:** You never instantiate this directly -- use a factory. But all your interactions with experiment data go through this class.

::: argos.experimentSetup.dataObjects.Experiment
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - refresh
        - setup
        - name
        - description
        - url
        - client
        - trialSet
        - entityType
        - entityTypeTable
        - entitiesTable
        - trialsTableAllSets
        - trialsTable
        - imageMap
        - getImage
        - getImageURL
        - getImageMetadata
        - getImageJSMappingFunction
        - getExperimentEntities
        - getEntitiesTypeByID
        - toJSON

---

### ExperimentZipFile

**Role:** Handles experiments packaged as `.zip` archives (the standard export format from ArgosWEB). Overrides `refresh()` to extract and version-migrate `data.json` from the archive, and `getImage()` to read images from inside the ZIP.

**Version migrations:** Automatically normalizes JSON schema versions 1.0.0, 2.0.0, and 3.0.0 to a standard internal format. See the [Architecture page](../architecture/experiment_setup.md#version-migration-detail) for migration details.

::: argos.experimentSetup.dataObjects.ExperimentZipFile
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - refresh
        - getImage

---

### TrialSet

**Role:** A named collection of trials (e.g., `"design"`, `"deploy"`). Extends `dict` so trials can be accessed by name: `trial_set["myTrial"]`. Defines the attribute type schema that all trials in the set share.

**Key concept:** The `attributeTypes` (accessed via `properties` / `propertiesTable`) define what properties each trial *can* have and their types. Individual trials then hold the actual values.

::: argos.experimentSetup.dataObjects.TrialSet
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - experiment
        - name
        - description
        - numberOfTrials
        - properties
        - propertiesTable
        - trials
        - trialsTable
        - toJSON

---

### Trial

**Role:** A single experimental configuration that maps entities to their trial-specific attributes. This is where per-experiment variables live (sensor positions, thresholds, calibration values).

**Key concept:** When you access `trial.entities` or `trial.entitiesTable`, the containment hierarchy is resolved: child entities inherit properties from their parents, locations are spread into lat/lon, and all property types are parsed. See the [containment resolution workflow](#call-workflow-accessing-trial-entity-data) above.

**Property parsing:** Each property value is parsed according to its type (defined in the parent TrialSet's `attributeTypes`): `location` → lat/lon/name, `number` → float, `boolean` → True/False, `datetime_local` → Timestamp with Israel timezone, etc.

::: argos.experimentSetup.dataObjects.Trial
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - experiment
        - trialSet
        - name
        - created
        - cloneFrom
        - numberOfEntities
        - properties
        - propertiesTable
        - entities
        - entitiesTable
        - toJSON

---

### EntityType

**Role:** Defines a category of devices or assets (e.g., `"Sensor"`, `"WeatherStation"`). Extends `dict` so entities can be accessed by name: `entity_type["Sensor_01"]`. Holds the attribute type schema (what properties entities of this type have) and the collection of entity instances.

**Key concept:** The `attributeTypes` include a `scope` field. Attributes with `scope="Constant"` become default values for all entities of this type. Other scopes are per-entity or per-trial.

::: argos.experimentSetup.dataObjects.EntityType
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - experiment
        - name
        - numberOfEntities
        - properties
        - propertiesTable
        - entitiesTable
        - entitiesAllProperties
        - toJSON

---

### Entity

**Role:** A single device or asset instance. Has constant properties (from its type defaults + its own attributes) and trial-specific properties (set per trial). Provides multiple views of its data: flat dict, list of dicts with scope, DataFrame, and per-trial lookups.

**Property scopes:**

- **Constant** -- from EntityType `attributeTypes` where `scope="Constant"` (type-level defaults)
- **Device** -- from the entity's own `attributes` in the metadata (instance-level overrides)
- **Trial** -- from trial entity data (changes per trial)

Use `entity.properties` for constant+device, `entity.trialProperties(set, trial)` for trial-specific, and `entity.allPropertiesTable` for everything combined.

::: argos.experimentSetup.dataObjects.Entity
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - name
        - entityType
        - experiment
        - properties
        - propertiesList
        - propertiesTable
        - allProperties
        - allPropertiesList
        - allPropertiesTable
        - allTrialProperties
        - allTrialPropertiesTable
        - trialProperties
        - toJSON

---

## Utility Functions

### Containment Resolution

**Role:** The `fillContained` module resolves parent-child entity relationships. When an entity is "contained in" another (e.g., a sensor on a pole at a location), it inherits the parent's properties (especially location). This is called automatically when you access `trial.entitiesTable`.

See the [containment resolution workflow](#call-workflow-accessing-trial-entity-data) above for the full call sequence.

::: argos.experimentSetup.fillContained.fill_properties_by_contained
    options:
      show_root_heading: true
      heading_level: 4

::: argos.experimentSetup.fillContained.spread_attributes
    options:
      show_root_heading: true
      heading_level: 4

::: argos.experimentSetup.fillContained.get_parent
    options:
      show_root_heading: true
      heading_level: 4

::: argos.experimentSetup.fillContained.key_from_name
    options:
      show_root_heading: true
      heading_level: 4
