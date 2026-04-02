"""
Data objects for the pyArgos experiment hierarchy.

This module defines the core data model: Experiment, TrialSet, Trial,
EntityType, and Entity. These objects are created by the factory classes
in ``dataObjectsFactory`` and provide a Pandas-based interface to
experiment metadata.
"""

import zipfile
import os
import json
import pandas
import matplotlib.pyplot as plt
import warnings

from argos.experimentSetup.fillContained import fill_properties_by_contained
from ..utils.jsonutils import loadJSON
from ..utils.logging import get_logger as argos_get_logger

import numpy


def testNan(x):
    """
    Test whether a value is NaN, handling non-numeric types gracefully.

    Parameters
    ----------
    x : any
        The value to test.

    Returns
    -------
    bool
        True if ``x`` is NaN, False otherwise (including for non-numeric types).
    """
    try:
        return numpy.isnan(x)
    except Exception:

        return False


class Experiment:
    """
    Interface to an Argos experiment.

    The Experiment object is the top-level container for all experiment data.
    It holds entity types, trial sets, and image maps, and exposes them as
    Pandas DataFrames for easy analysis.

    Experiments can be loaded from a JSON file, a dict, or via the factory
    classes (``fileExperimentFactory``, ``webExperimentFactory``).

    Parameters
    ----------
    setupFileOrData : str or dict
        Either a file path to a JSON file containing the experiment
        configuration, or a dict with the experiment data already loaded.

    Examples
    --------
    >>> from argos.experimentSetup import fileExperimentFactory
    >>> experiment = fileExperimentFactory("/path/to/experiment").getExperiment()
    >>> print(experiment.name)
    >>> print(experiment.entitiesTable)
    """

    _setupFileNameOrData = None  # experiment description holds its name, descrition and ect.
    _experimentSetup = None

    _trialSetsDict = None  # A dictionary of the trial sets.
    _entitiesTypesDict = None  # A dictionary of the devices types.

    _imagesMap = None

    def refresh(self):
        """
        Reload the experiment data from the source and rebuild all internal structures.

        Re-reads the setup file or dict, then re-initializes all trial sets
        and entity types from the fresh data.
        """

        self._experimentSetup = loadJSON(self._setupFileNameOrData)
        self._initTrialSets()
        self._initEntitiesTypes()

    @property
    def setup(self):
        """
        The raw experiment configuration dictionary.

        Returns
        -------
        dict
            The full experiment setup as loaded from the source file or dict.
        """
        return self._experimentSetup

    @property
    def name(self):
        """
        The name of the experiment.

        Handles different JSON structures across versions:

        - v3.0.0 ZIP (after migration): ``setup['experiment']['name']``
        - v2.0.0 ZIP (after migration): ``setup['experiment']['name']``
        - Legacy/direct: ``setup['name']``

        Returns
        -------
        str
            The experiment name as defined in the configuration.
        """
        return self._get_experiment_field('name')

    @property
    def description(self):
        """
        The description of the experiment.

        Handles different JSON structures across versions.
        See :attr:`name` for the lookup order.

        Returns
        -------
        str
            The experiment description as defined in the configuration.
        """
        return self._get_experiment_field('description')

    def _get_experiment_field(self, field):
        """
        Look up an experiment-level field across different JSON structures.

        Checks in order:

        1. ``setup[field]`` (legacy/direct)
        2. ``setup['experiment'][field]`` (v2/v3 ZIP after migration)

        Parameters
        ----------
        field : str
            The field name (e.g., ``'name'``, ``'description'``).

        Returns
        -------
        any
            The field value.

        Raises
        ------
        KeyError
            If the field is not found in any of the expected locations.
        """
        if field in self.setup:
            return self.setup[field]
        if 'experiment' in self.setup and field in self.setup['experiment']:
            return self.setup['experiment'][field]
        raise KeyError(f"Field '{field}' not found in experiment setup")

    @property
    def trialSet(self):
        """
        Dictionary of all trial sets in the experiment.

        Returns
        -------
        dict[str, TrialSet]
            A dictionary mapping trial set names to ``TrialSet`` objects.
            Access individual trials via ``experiment.trialSet["design"]["myTrial"]``.
        """
        return self._trialSetsDict

    @property
    def entityType(self):
        """
        Dictionary of all entity types in the experiment.

        Returns
        -------
        dict[str, EntityType]
            A dictionary mapping entity type names to ``EntityType`` objects.
            Access individual entities via ``experiment.entityType["Sensor"]["Sensor_01"]``.
        """
        return self._entitiesTypesDict

    @property
    def entityTypeTable(self):
        """
        Summary table of all entity types and their properties.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per property per entity type,
            with an ``entityType`` column identifying the type.
        """
        entityTypeList = []
        for entityTypeName, entityTypeData in self.entityType.items():
            entityTypeList.append(entityTypeData.propertiesTable.assign(entityType=entityTypeName))

        return pandas.concat(entityTypeList, ignore_index=True)

    @property
    def entitiesTable(self):
        """
        Flat table of all entities across all entity types.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per entity, including the entity's constant
            properties plus ``entityType`` and ``entityName`` columns.
        """
        entityTypeList = []
        for entityTypeName, entityTypeData in self.entityType.items():
            for entityTypeDataName, entityData in entityTypeData.items():
                entityTypeList.append(entityData.propertiesTable.assign(entityType=entityTypeName,entityName=entityData.name))

        return pandas.concat(entityTypeList, ignore_index=True)

    @property
    def trialsTableAllSets(self):
        """
        Combined table of all trials across all trial sets.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per trial, with a ``trialSet`` column
            identifying which trial set each trial belongs to.
        """
        tableList = []
        for trialSetName in self.trialSet:
            tableList.append(self.trialsTable(trialSetName).assign(trialSet=trialSetName))
        return pandas.concat(tableList)


    def trialsTable(self, trialsetName):
        """
        Get the trials table for a specific trial set.

        Parameters
        ----------
        trialsetName : str
            The name of the trial set.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per trial in the specified trial set.
        """
        return self.trialSet[trialsetName].trialsTable

    def __init__(self, setupFileOrData):
        """
        Initialize the Experiment from a file path or dict.

        Parameters
        ----------
        setupFileOrData : str or dict
            Either a path to a JSON file containing the experiment
            configuration (downloaded from ArgosWEB), or a dict with
            the experiment data already parsed.
        """
        self.logger = argos_get_logger(self)
        self._trialSetsDict = dict()
        self._entitiesTypesDict = dict()

        self._setupFileNameOrData = setupFileOrData
        self.refresh()

        self.logger.execution("Loading images")
        self._init_ImageMaps()

    def _init_ImageMaps(self):
        """Initialize the images map. Overridden by ExperimentZipFile."""
        self._imagesMap = dict()

    @property
    def imageMap(self):
        """
        Dictionary of all image maps in the experiment.

        Image maps are used for spatial visualization of entities on
        experiment site images.

        Returns
        -------
        dict
            A dictionary mapping image names to their metadata (URL,
            bounding coordinates, dimensions).
        """
        return self._imagesMap

    def getImageURL(self, imageName: str):
        """
        Get the full URL of an experiment image.

        Parameters
        ----------
        imageName : str
            The name of the image.

        Returns
        -------
        str
            The full URL to the image file.
        """
        return self._imagesMap[imageName]['imageURL']

    def getImageJSMappingFunction(self, imageName: str):
        """
        Return a JavaScript mapping function for image coordinate transformation.

        Generates JavaScript code that maps real-world coordinates to normalized
        (0..1) image coordinates. Used in ThingsBoard dashboards for overlaying
        entity positions on map images.

        Parameters
        ----------
        imageName : str
            The name of the image.

        Returns
        -------
        str
            A JavaScript code string that takes ``origXPos`` and ``origYPos``
            and returns ``{x: image_x, y: image_y}`` in normalized coordinates.
        """
        metadata = self.getImageMetadata(imageName)

        xdim = metadata['right'] - metadata['left']
        ydim = metadata['upper'] - metadata['lower']

        Xconvert = f"image_x = (origXPos - ({metadata['left']}))/{xdim};"
        Yconvert = f"image_y = (({metadata['upper']})-origYPos)/{ydim};"

        return_string = "return {x: image_x, y: image_y};"

        return "\n".join([Xconvert, Yconvert, return_string])

    def getImageMetadata(self, imageName: str):
        """
        Get the metadata for an experiment image.

        Parameters
        ----------
        imageName : str
            The name of the image.

        Returns
        -------
        dict
            A dictionary containing the image metadata including bounding
            coordinates (``left``, ``right``, ``upper``, ``lower``),
            dimensions (``width``, ``height``), and the image URL.
        """
        return self._imagesMap[imageName]

    def _initTrialSets(self):

        for trialset in self.setup['trialSets']:
            self._trialSetsDict[trialset['name']] = TrialSet(experiment=self, metadata=trialset)

    def _initEntitiesTypes(self):
        for entityType in self.setup['entityTypes']:
            self._entitiesTypesDict[entityType['name']] = EntityType(experiment=self, metadata=entityType)

    def toJSON(self):
        """
        Export the entire experiment to a nested dictionary.

        Serializes entity types, trial sets, and experiment metadata
        into a single JSON-compatible dict.

        Returns
        -------
        dict
            A dictionary with keys ``entityType``, ``trialSet``, and
            ``experiment`` containing all experiment data.
        """
        ret = dict()

        entityTypeMap = dict()
        for entityName, entityType in self.entityType.items():
            entityTypeMap[entityName] = entityType.toJSON()

        trialMap = dict()
        for trialName, trialData in self.trialSet.items():
            trialMap[trialName] = trialData.toJSON()

        ret['entityType'] = entityTypeMap
        ret['trialSet'] = trialMap

        expr = dict()

        for field in ['maps', 'begin', 'end', 'description']:
            expr[field] = self._setupFileNameOrData['experimentsWithData'][field]

        ret['experiment'] = expr
        return ret

    def getExperimentEntities(self):
        """
        Get a flat list of all entities in the experiment.

        Returns
        -------
        list[dict]
            A list of dicts, each with ``entityName`` and ``entityTypeName`` keys.
        """
        retList = []

        for entitytypeName, entityTypeObj in self.entityType.items():
            for entityName, entityData in entityTypeObj.items():
                retList.append(dict(entityName=entityName, entityTypeName=entityData.entityType.name))

        return retList

    def getEntitiesTypeByID(self, entityTypeID):
        """
        Look up an entity type by its internal key ID.

        Parameters
        ----------
        entityTypeID : str
            The key ID of the entity type.

        Returns
        -------
        EntityType or None
            The matching EntityType object, or None if not found.
        """

        ret = None
        for entityTypeName, entityTypeData in self.entityType.items():
            if entityTypeID == entityTypeData.keyID:
                ret = entityTypeData
                break

        return ret

    def getImage(self, imageName: str):
        """
        Load an experiment image.

        Override in subclasses to provide image loading from the
        appropriate source (e.g., ZIP archive).

        Parameters
        ----------
        imageName : str
            The name of the image.

        Returns
        -------
        numpy.ndarray
            The image data as a NumPy array.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError("getImage must be called on ExperimentZipFile, not base Experiment")


class ExperimentZipFile(Experiment):
    """
    Experiment loaded from a ZIP archive.

    Extends :class:`Experiment` to handle experiments packaged as ``.zip``
    files (as downloaded from ArgosWEB). The ZIP archive is expected to
    contain a ``data.json`` file and an ``images/`` directory.

    Handles automatic version migration for JSON schema versions 1.0.0,
    2.0.0, and 3.0.0.

    Parameters
    ----------
    setupFileOrData : str
        Path to the ``.zip`` file containing the experiment data.
    """

    def __init__(self, setupFileOrData):
        """
        Initialize ExperimentZipFile from a ZIP archive path.

        Parameters
        ----------
        setupFileOrData : str
            Path to the ``.zip`` file containing the experiment data.
        """
        super().__init__(setupFileOrData=setupFileOrData)

    def getImage(self, imageName: str):
        """
        Load an image from inside the ZIP archive.

        Parameters
        ----------
        imageName : str
            The name of the image.

        Returns
        -------
        numpy.ndarray
            The image data as a NumPy array (via matplotlib.pyplot.imread).
        """

        imageurl = self._imagesMap[imageName]['filename']

        try:
            # For compliance with the old version.
            with zipfile.ZipFile(self._setupFileNameOrData) as archive:
                imageFile = archive.open(os.path.join("images", imageurl))
        except KeyError:
            # This is the new version
            with zipfile.ZipFile(self._setupFileNameOrData) as archive:
                imageFile = archive.open(os.path.join("images", imageName+".png"))

        img = plt.imread(imageFile)
        imageFile.close()
        return img

    def refresh(self):
        """
        Reload the experiment data from the ZIP archive.

        Extracts ``data.json`` from the archive, detects the JSON schema
        version, applies the appropriate migration, and rebuilds all
        trial sets and entity types.
        """
        self.logger.execution("------- Start ----")
        self.logger.debug(f"Loading file {self._setupFileNameOrData}")

        with zipfile.ZipFile(self._setupFileNameOrData) as archive:
            experimentDict = loadJSON("\n".join([x.decode() for x in archive.open("data.json").readlines()]))

        fileVersion = experimentDict.get("version", "1.0.0.").replace(".", "_")
        self.logger.debug(f"Got file version {fileVersion}")

        experimentDict = getattr(self, f"_fix_json_version_{fileVersion}")(experimentDict)

        self.logger.execution("Experiemnt dict")
        self._experimentSetup = experimentDict

        self.logger.execution("Init trial sets")
        self._initTrialSets()

        self.logger.execution("Init entity type")
        self._initEntitiesTypes()

    def _init_ImageMaps(self):
        ## Initializing the images map
        #with zipfile.ZipFile(self._setupFileNameOrData) as archive:
        #    experimentDict = loadJSON("\n".join([x.decode() for x in archive.open("data.json").readlines()]))
        experimentDict = self.setup

        self._imagesMap = dict()
        for imgs in experimentDict.get('maps',[]):
            imgName = imgs['name']
            self._imagesMap[imgName] = imgs

    def _fix_json_version_1_0_0_(self, jsonFile):
        """Pass through for version 1.0.0 (no migration needed)."""
        return jsonFile

    def _fix_json_version_2_0_0_(self, jsonFile):
        """Migrate version 2.0.0 JSON to the standard internal format."""

        oldFormat = dict(experiment=jsonFile['experiment'],
                         entityTypes=jsonFile['entityTypes'],
                         trialSets=jsonFile['trialSets'])

        for trialSet in oldFormat['trialSets']:
            trialSet['trials'] = []
            currentKey = trialSet['key']
            for trial in jsonFile['trials']:
                if trial['trialSetKey'] == currentKey:
                    trialSet['trials'].append(trial)

        for entityType in oldFormat['entityTypes']:
            entityType['entities'] = []
            currentKey = entityType['key']
            for entity in jsonFile['entities']:
                if entity['entitiesTypeKey'] == currentKey:
                    entityType['entities'].append(entity)

        return oldFormat

    def _fix_json_version_3_0_0(self, jsonFile):
        """Migrate version 3.0.0 JSON to the standard internal format."""
        oldFormat = dict(experiment={'name': jsonFile['name'],
                                     'description': jsonFile.get('description', ''),
                                     'version': jsonFile['version'],
                                     'startDate': jsonFile['startDate'],
                                     'endDate': jsonFile['endDate']},
                         entityTypes=jsonFile.get('deviceTypes', []),
                         trialSets=jsonFile.get('trialTypes', []),
                         maps = jsonFile.get("imageStandalone",[]),
                         shapes=jsonFile.get("shapes",[]),
                         )

        for entityType in oldFormat['entityTypes']:
            entityType['entities'] = []
            for device in entityType.get('devices', []):
                entityType['entities'].append(device)

        for trialSet in oldFormat['trialSets']:
            for trial in trialSet.get('trials', []):
                if 'properties' not in trial.keys():
                    trial['properties'] = []
                if 'state' not in trial.keys():
                    trial['state'] = None

                trial['entities'] = trial.get('devicesOnTrial', [])

        return oldFormat


class TrialSet(dict):
    """
    A collection of trials belonging to a single trial set.

    Extends ``dict`` so that individual trials can be accessed by name
    using dictionary syntax: ``trial_set["myTrial"]``.

    A trial set groups related trials together (e.g., "design" for planned
    trials, "deploy" for active deployments). Each trial set has its own
    attribute type definitions that determine what properties each trial
    can have.

    Parameters
    ----------
    experiment : Experiment
        The parent experiment object.
    metadata : dict
        The trial set metadata from the experiment configuration,
        including name, description, attribute types, and trials.
    """
    _experiment = None
    _metadata = None

    _trialsDict = None

    @property
    def experiment(self):
        """
        The parent experiment object.

        Returns
        -------
        Experiment
            The experiment this trial set belongs to.
        """
        return self._experiment

    @property
    def name(self):
        """
        The name of the trial set.

        Returns
        -------
        str
            The trial set name (e.g., ``"design"``, ``"deploy"``).
        """
        return self._metadata['name']

    @property
    def description(self):
        """
        The description of the trial set.

        Returns
        -------
        str
            A human-readable description of the trial set's purpose.
        """
        return self._metadata['description']

    @property
    def numberOfTrials(self):
        """
        The number of trials in this trial set.

        Returns
        -------
        int
            The count of trials.
        """
        return len(self._metadata)

    @property
    def propertiesTable(self):
        """
        The attribute type definitions as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame describing the attribute types (columns: key, type,
            label, description, etc.) that trials in this set can have.
        """
        return pandas.DataFrame(self.properties)

    @property
    def properties(self):
        """
        The raw attribute type definitions for this trial set.

        Returns
        -------
        list[dict]
            A list of attribute type dicts, each with keys like ``key``,
            ``type``, ``label``, ``description``, ``required``, ``trialField``.
        """
        return self._metadata['attributeTypes']

    def toJSON(self):
        """
        Export the trial set to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary with ``name``, ``properties``, and ``trials`` keys.
            The ``trials`` value is a dict mapping trial names to their
            serialized data.
        """
        ret = dict()
        ret['name'] = self.name
        ret['properties'] = self.properties

        trialsJSON = {}
        for trialName, trialData in self.items():
            trialsJSON[trialName] = trialData.toJSON()

        ret['trials'] = trialsJSON

        return ret

    @property
    def trials(self):
        """
        Dictionary of all trials in this set, serialized as dicts.

        Returns
        -------
        dict[str, dict]
            A dictionary mapping trial names to their serialized data.
        """
        return self.toJSON()['trials']

    @property
    def trialsTable(self):
        """
        Table of all trials in this set.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per trial, indexed by trial name,
            containing each trial's properties.
        """
        return pandas.DataFrame(self.toJSON()['trials']).T

    def __init__(self, experiment: Experiment, metadata: dict):
        """
        Initialize a TrialSet.

        Parameters
        ----------
        experiment : Experiment
            The parent experiment object.
        metadata : dict
            The trial set metadata from the experiment configuration,
            including name, description, attribute types, and trials.
        """
        self._experiment = experiment
        self._metadata = metadata
        self._initTrials()

    def _initTrials(self):

        for trial in self._metadata.get('trials', []):
            self[trial['name']] = Trial(trialSet=self, metadata=trial)


class Trial:
    """
    A single experimental trial within a trial set.

    A trial represents a specific experimental configuration, mapping
    entities to their trial-specific attributes (e.g., sensor locations,
    thresholds, calibration values).

    Trials parse their properties using type-specific handlers for
    location, text, number, boolean, datetime, and selectList types.

    Parameters
    ----------
    trialSet : TrialSet
        The parent trial set.
    metadata : dict
        The trial metadata from the experiment configuration.
    """

    _trialSet = None
    _metadata = None

    @property
    def experiment(self):
        """
        The root experiment object.

        Returns
        -------
        Experiment
            The experiment this trial belongs to (via the parent trial set).
        """
        return self._trialSet.experiment

    @property
    def experiment(self):
        """
        The root experiment object.

        Returns
        -------
        Experiment
            The experiment this trial belongs to (via the parent trial set).
        """
        return self._trialSet.experiment

    @property
    def name(self):
        """
        The name of the trial.

        Returns
        -------
        str
            The trial name as defined in the configuration.
        """
        return self._metadata['name']

    @property
    def created(self):
        """
        The creation timestamp of the trial.

        Returns
        -------
        str
            The ISO timestamp string of when the trial was created.
        """
        return self._metadata['created']


    @property
    def cloneFrom(self):
        """
        The name of the trial this was cloned from, if any.

        Returns
        -------
        str or None
            The source trial name, or None if this is an original trial.
        """
        return self._metadata['cloneFrom']

    @property
    def numberOfEntities(self):
        """
        The number of entities associated with this trial.

        Returns
        -------
        int
            The count of entities in the trial metadata.
        """
        return len(self._metadata)

    @property
    def properties(self):
        """
        The trial-level properties as a parsed dictionary.

        Property values are parsed according to their type (text, number,
        location, boolean, datetime, selectList).

        Returns
        -------
        dict
            A dictionary mapping property labels to their parsed values.
        """

        propDict = {}
        propList = self._metadata['attributes'] if 'attributes' in self._metadata else self._metadata['properties']

        for prop in propList:
            propDict[prop['name']] = prop['value']

        return propDict

    @property
    def trialSet(self):
        """
        The parent trial set.

        Returns
        -------
        TrialSet
            The trial set this trial belongs to.
        """
        return self._trialSet

    @property
    def propertiesTable(self):
        """
        The trial-level properties as a single-row DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row containing all trial properties
            as columns.
        """
        return pandas.DataFrame(self.properties,index=[0])

    def __init__(self, trialSet: TrialSet, metadata: dict):
        """
        Initialize a Trial.

        Parses the trial's properties by merging raw property values with
        the trial set's attribute type definitions, then applying type-specific
        parsers for each property.

        Parameters
        ----------
        trialSet : TrialSet
            The parent trial set object.
        metadata : dict
            The trial metadata dict containing name, created, cloneFrom,
            properties, and entities.
        """
        self._trialSet = trialSet
        self._metadata = metadata
        if len(metadata.get('properties', [])):
            propertiesPandas = pandas.DataFrame(metadata['properties']).set_index('key')

            properties = propertiesPandas.merge(trialSet.propertiesTable, left_index=True, right_index=True)[
                ['val', 'type', 'label', 'description']]
            getParser = lambda x: getattr(self, f"_parseProperty_{x.replace('-', '_')}")

            #       this wont work well for location property in the trial because it has 2 fields.
            #       we have to get the list, and the change all the lists with size 1 to the object itself (like we do now)
            #       and leave all the lists with size 2 as is.

            parsedValuesList = []
            for key, data in properties.T.to_dict().items():
                parsed_data = getParser(data['type'])(data, data)
                if len(parsed_data[1]) > 0:
                    val = parsed_data[1][0]
                else:
                    val = None
                parsedValuesList.append((data['label'], val))

            self._properties = dict(parsedValuesList)
        else:
            self._properties = dict()

    def toJSON(self):
        """
        Export the trial to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary of the trial's properties with an added ``name`` key.
        """
        val = self.properties
        val['name'] = self.name
        return val

    def __str__(self):
        """Return the JSON string representation of the trial."""
        return json.dumps(self.toJSON())

    def __repr__(self):
        """Return a compact JSON string with just the trial name."""
        return json.dumps(dict(name=self.name))

    def _parseProperty_location(self, property, propertyMetadata):
        """
        Parse a location property into name, latitude, and longitude.

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing location data
            (either a dict or a JSON string with ``name`` and ``coordinates``).
        propertyMetadata : dict
            The attribute type metadata for this property.

        Returns
        -------
        tuple[list[str], list]
            A tuple of (column_names, values) where column_names is
            ``['locationName', 'latitude', 'longitude']`` and values
            are the corresponding parsed values. Returns empty lists
            if the location data is missing or malformed.
        """
        try:
            if isinstance(property['val'], dict):
                locationDict = property['val']
            else:
                locationDict = json.loads(property['val'])

            locationName = locationDict['name']
            coords = locationDict['coordinates']
            if isinstance(coords, str):
                coords = eval(coords)

            latitude = float(coords[0])
            longitude = float(coords[1])
            data = [locationName, latitude, longitude]
            columns = ['locationName', 'latitude', 'longitude']
        except KeyError:
            # data = [None] * 3
            data = []
            columns = []
        except TypeError:
            # data = [None] * 3
            data = []
            columns = []

        return columns, data

    def _parseProperty_text(self, property, propertyMetadata):
        """
        Parse a text property.

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing the text value.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list]
            A tuple of (column_names, values) with the property label
            and its string value.
        """
        propertyLabel = propertyMetadata['label']
        data = [property['val']]
        columns = [propertyLabel]

        return columns, data

    def _parseProperty_textArea(self, property, propertyMetadata):
        """
        Parse a text area property (multi-line text).

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing the text value.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list]
            A tuple of (column_names, values) with the property label
            and its string value.
        """
        propertyLabel = propertyMetadata['label']

        data = [property['val']]
        columns = [propertyLabel]

        return columns, data

    def _parseProperty_boolean(self, property, propertyMetadata):
        """
        Parse a boolean property.

        Converts string values ``'true'``, ``'yes'``, and ``'1'``
        (case-insensitive) to ``True``; all other values to ``False``.

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing the boolean string.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list[bool]]
            A tuple of (column_names, values) with the property label
            and a Python boolean.
        """
        propertyLabel = propertyMetadata['label']
        data = [property['val'].lower() in ['true','yes','1']]
        columns = [propertyLabel]

        return columns, data

    def _parseProperty_number(self, property, propertyMetadata):
        """
        Parse a numeric property to float.

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing the numeric string.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list[float]]
            A tuple of (column_names, values) with the property label
            and a float value. Returns empty lists if the value cannot
            be converted to float.
        """
        try:
            propertyLabel = propertyMetadata['label']
            flt = property['val']
            if flt is None:
                data = [None]
            else:
                data = [float(property['val'])]
            columns = [propertyLabel]
        except ValueError:
            # print(f"\tCannot convert to float property {propertyLabel}. Got value '{property['val']}'")
            data = []
            columns = []

        return columns, data

    def _parseProperty_datetime_local(self, property, propertyMetadata):
        """
        Parse a local datetime property with Israel timezone.

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing a datetime string.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list[pandas.Timestamp]]
            A tuple of (column_names, values) with the property label
            and a timezone-aware Timestamp (Israel timezone).
        """
        propertyLabel = propertyMetadata['label']
        localdata = pandas.to_datetime(property['val'], utc=False)
        if localdata is None:
            data = [None]
        else:
            data = [localdata.tz_localize("israel")]
        columns = [propertyLabel]

        return columns, data

    def _parseProperty_selectList(self, property, propertyMetadata):
        """
        Parse a select list property (value chosen from predefined options).

        Parameters
        ----------
        property : dict
            The property dict with a ``val`` key containing the selected value.
        propertyMetadata : dict
            The attribute type metadata (used for the ``label``).

        Returns
        -------
        tuple[list[str], list]
            A tuple of (column_names, values) with the property label
            and the selected value.
        """
        propertyLabel = propertyMetadata['label']
        data = [property['val']]
        columns = [propertyLabel]

        return columns, data

    def _composeEntityProperties(self, entityType, properties):
        """
        Resolve property names and parse values for a single entity's trial properties.

        Splits location properties into separate latitude/longitude columns.

        Parameters
        ----------
        entityType : EntityType
            The entity type (used to look up property metadata).
        properties : list[dict]
            List of property dicts with ``key`` and ``val`` keys.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with parsed property values as columns.
        """
        data = []
        columns = []
        for property in properties:
            propertyKey = property['key']
            propertyMetadata = entityType.propertiesTable.loc[propertyKey]
            propertyType = propertyMetadata['type']

            prop_type_handler = getattr(self, f"_parseProperty_{propertyType}")

            pcolumns, pdata = prop_type_handler(property, propertyMetadata)
            columns += pcolumns
            data += pdata

        entity_trial_properties = pandas.DataFrame(data=[data], columns=columns, index=[0])
        return entity_trial_properties

    def _composeProperties(self, entities):
        """
        Compose the full properties table for trial entities.

        Joins trial-specific properties with entity constant properties
        for all entities in the trial.

        Parameters
        ----------
        entities : pandas.DataFrame
            A DataFrame of entity references from the trial metadata.

        Returns
        -------
        pandas.DataFrame
            A combined DataFrame with both trial-specific and constant
            properties for each entity.
        """
        if 'entitiesTypeKey' in entities.columns:
            fullData = self.experiment._entitiesTableFull.set_index("key").join(entities, rsuffix="_r",
                                                                                how="inner").reset_index()

            dfList = []
            for indx, (entitykey, entitydata) in enumerate(fullData.iterrows()):
                properties = entitydata['properties']
                entityType = self.experiment.getEntitiesTypeByID(entityTypeID=entitydata.entitiesTypeKey)

                entity_trial_properties = self._composeEntityProperties(entityType, properties)

                entityProperties = entityType[entitydata['name']].propertiesTable.copy()
                entity_total_properties = entity_trial_properties.join(entityProperties,
                                                                       how='left',
                                                                       rsuffix='_prop')  # .assign(trialSet = self.trialSet.name,

                dfList.append(entity_total_properties)
            new_df = pandas.concat(dfList, sort=False, ignore_index=True).drop(columns=["key", "entitiesTypeKey"])
        else:
            new_df = pandas.DataFrame(entities)

        return new_df

    @property
    def entities(self):
        """
        The trial's entity data as a dictionary.

        Returns entity names mapped to their properties, with NaN values
        filtered out.

        Returns
        -------
        dict[str, dict]
            A dictionary mapping entity names to their property dicts.
            Returns an empty dict if no entities are associated with
            this trial.
        """
        ret = self.entitiesTable
        if ret.empty:
            return dict()
        else:
            datadict = ret.set_index("deviceItemName").T.to_dict()
            resultProperties = dict()
            for entityName, entityData in datadict.items():
                resultProperties[entityName] = dict(
                    [(propName, propData) for propName, propData in entityData.items() if not testNan(propData)])
            return resultProperties


    @property
    def entitiesTable(self):
        """
        The trial's entity data as a DataFrame.

        Resolves containment hierarchies (parent-child entity relationships)
        and composes the full properties for each entity.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per entity, containing both trial-specific
            and inherited properties.
        """
        filled_entities = fill_properties_by_contained(self._trialSet.experiment.entityType, self._metadata['entities'])
        if len(filled_entities) == 0:
            entities = pandas.DataFrame()
        elif 'key' in filled_entities[0].keys():
            entities = pandas.DataFrame(filled_entities).set_index('key')
        else:
            entities = pandas.DataFrame(filled_entities)
        return self._composeProperties(entities)

    def _prepareEntitiesMetadata(self, metadata):
        """
        Flatten entity metadata into a DataFrame for processing.

        Parameters
        ----------
        metadata : list[dict]
            Raw entity metadata from the trial configuration.

        Returns
        -------
        pandas.DataFrame
            A flattened DataFrame with entity properties and type keys.
        """

        retList = []
        for entityData in metadata:
            for propData in entityData['properties']:
                properties = pandas.DataFrame(propData, index=[0])
                itm = pandas.DataFrame(properties).assign(entitiesTypeKey=entityData['entitiesTypeKey'],
                                                          containsEntities=entityData['containsEntities'])
                retList.append(itm)

        return pandas.concat(retList, ignore_index=True)


################################### Depreacted part of trial

    @property
    def deployEntitiesTable(self):
        """
        .. deprecated::
            Use :attr:`entitiesTable` instead.
        """
        warnings.warn("deployEntitiesTable is deprecated. Use entitiesTable", DeprecationWarning, stacklevel=2)
        return self.entitiesTable

    @property
    def designEntities(self):
        """
        .. deprecated::
            Use :attr:`entities` instead.
        """
        warnings.warn("designEntities is deprecated. Use entities", DeprecationWarning, stacklevel=2)
        return self.entities

    @property
    def designEntitiesTable(self):
        """
        .. deprecated::
            Use :attr:`entitiesTable` instead.
        """
        warnings.warn("designEntitiesTable is deprecated. Use entitiesTable", DeprecationWarning, stacklevel=2)
        return self.entitiesTable

    @property
    def deployEntities(self):
        """
        .. deprecated::
            Use :attr:`entities` instead.
        """
        warnings.warn("deployEntities is deprecated. Use entitiesTable", DeprecationWarning, stacklevel=2)
        return self.entities

##################################################################


class EntityType(dict):
    """
    A collection of entities of the same type.

    Extends ``dict`` so individual entities can be accessed by name:
    ``entity_type["Sensor_01"]``.

    An EntityType defines the attribute schema (property types) that all
    entities of this type share, and holds the collection of individual
    Entity instances.

    Parameters
    ----------
    experiment : Experiment
        The parent experiment object.
    metadata : dict
        The entity type metadata from the experiment configuration,
        including name, attribute types, and entities.
    """
    _experiment = None
    _metadata = None


    def __init__(self, experiment: Experiment, metadata: dict):
        """
        Initialize an EntityType.

        Parameters
        ----------
        experiment : Experiment
            The parent experiment object.
        metadata : dict
            The entity type metadata including ``name``, ``attributeTypes``,
            and ``entities`` list.
        """
        self._experiment = experiment
        self._metadata = metadata

        self._initEntities()

    def _initEntities(self):
        for entity in self._metadata.get('entities', []):
            self[entity['name']] = Entity(entityType=self, metadata=entity)

    @property
    def experiment(self):
        """
        The parent experiment object.

        Returns
        -------
        Experiment
            The experiment this entity type belongs to.
        """
        return self._experiment

    @property
    def name(self):
        """
        The name of the entity type.

        Returns
        -------
        str
            The type name (e.g., ``"Sensor"``, ``"WeatherStation"``).
        """
        return self._metadata['name']

    @property
    def numberOfEntities(self):
        """
        The number of entities of this type.

        Returns
        -------
        int
            The count of entities in this type.
        """
        return len(self)

    @property
    def propertiesTable(self):
        """
        The attribute type definitions as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame describing the attribute types (key, type, label,
            description, etc.) that entities of this type can have.
            Indexed by the attribute key.
        """
        return pandas.DataFrame(self.properties)

    @property
    def properties(self):
        """
        The raw attribute type definitions for this entity type.

        Returns
        -------
        list[dict]
            A list of attribute type dicts, each with keys like ``key``,
            ``type``, ``label``, ``description``, ``scope``, ``defaultValue``.
        """
        return self._metadata['attributeTypes']

    def toJSON(self):
        """
        Export the entity type to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary with ``name``, ``properties``, and ``entities`` keys.
            The ``entities`` value maps entity names to their serialized data.
        """
        ret = dict()
        ret['name'] = self.name
        ret['properties'] = self.properties

        entityJSON = {}
        for entityName, entityData in self.items():
            entityJSON[entityName] = entityData.toJSON()

        ret['entities'] = entityJSON

        return ret


    @property
    def entitiesTable(self):
        """
        Table of all entities of this type with their constant properties.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per entity, indexed by ``entityName``,
            containing each entity's constant (non-trial) properties.
        """
        retList = []
        for entityName, entityData in self.items():
            trialProps = entityData.propertiesTable.assign(entityName=entityName).reset_index(drop=True)
            retList.append(trialProps)
        return pandas.concat(retList).set_index("entityName")

    @property
    def entitiesAllProperties(self):
        """
        Table of all entities with all properties, including trial-specific ones.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with a multi-level index of (``entityName``, ``trialName``),
            containing both constant and trial-specific properties.
        """
        retList = []
        for entityName, entityData in self.items():
            trialProps = entityData.allPropertiesTable.assign(entityName=entityName).reset_index(drop=True)
            retList.append(trialProps)
        return pandas.concat(retList).set_index(["entityName","trialName"])


class Entity:
    """
    A single device or asset instance in the experiment.

    An Entity represents one physical or logical device/asset. It has
    constant properties (defined by its EntityType) and trial-specific
    properties (set per trial).

    Parameters
    ----------
    entityType : EntityType
        The parent entity type.
    metadata : dict
        The entity metadata including ``name`` and ``attributes``.
    """
    _entityType = None
    _metadata = None

    def __init__(self, entityType: EntityType, metadata: dict):
        """
        Initialize an Entity.

        Collects constant-scope properties from the entity type's attribute
        definitions and device-scope properties from the entity's own
        attributes.

        Parameters
        ----------
        entityType : EntityType
            The parent entity type object.
        metadata : dict
            The entity metadata dict containing ``name`` and optionally
            ``attributes`` (a list of ``{name, value}`` dicts).
        """
        self._entityType = entityType
        self._metadata = metadata

        self._properties = []
        for attr in self._entityType._metadata['attributeTypes']:
            if attr.get("scope","") == 'Constant':
                self._properties.append(dict(name=attr['name'],value=attr['defaultValue'],scope="Constant"))

        for attr in metadata.get('attributes',[]):
            self._properties.append(dict(name=attr['name'], value=attr['value'],scope="Device"))


    @property
    def entityType(self):
        """
        The name of this entity's type.

        Returns
        -------
        str
            The entity type name (e.g., ``"Sensor"``).
        """
        return self._entityType.name

    @property
    def experiment(self):
        """
        The root experiment object.

        Returns
        -------
        Experiment
            The experiment this entity belongs to (via the parent entity type).
        """
        return self._entityType.experiment

    @property
    def name(self):
        """
        The name of this entity.

        Returns
        -------
        str
            The entity name (e.g., ``"Sensor_01"``).
        """
        return self._metadata['name']

    @property
    def properties(self):
        """
        The entity's constant properties as a dictionary.

        Returns
        -------
        dict[str, any]
            A dictionary mapping property names to their values.
            Includes both Constant-scope (from type defaults) and
            Device-scope (from entity attributes) properties.
        """
        return {prop['name'] : prop['value'] for prop in self._properties}

    @property
    def propertiesList(self):
        """
        The entity's constant properties as a list of dicts.

        Returns
        -------
        list[dict]
            A list of property dicts, each with ``name``, ``value``,
            and ``scope`` keys.
        """
        return self._properties

    @property
    def allProperties(self):
        """
        All properties including trial-specific ones, organized hierarchically.

        Returns
        -------
        dict[str, dict[str, dict]]
            A nested dictionary: ``{trialSetName: {trialName: {trialName: properties}}}``.
        """
        trialsetdict = dict() #$self.propertiesList
        for trialsSetsName in self.experiment.trialSet.keys():
            trialsetdict[trialsSetsName] = dict()
            for trialName in self.experiment.trialSet[trialsSetsName].keys():
                ddp = trialsetdict[trialsSetsName].setdefault(trialName, dict())
                ddp[trialName] = self.trialProperties(trialsSetsName, trialName)

        return trialsetdict

    @property
    def allPropertiesList(self):
        """
        All properties as a flat list, including trial-specific ones.

        Returns
        -------
        list[dict]
            A list combining constant properties with trial-specific
            properties. Trial entries include ``trialSetName``,
            ``trialName``, and ``scope='trial'`` keys.
        """
        trialsetlist = self.propertiesList
        for trialsSetsName in self.experiment.trialSet.keys():
            for trialName in self.experiment.trialSet[trialsSetsName].keys():
                deploy = self.trialProperties(trialsSetsName, trialName)
                deploy['trialSetName'] = trialsSetsName
                deploy['trialName'] = trialName
                deploy['scope'] = 'trial'
                trialsetlist.append(deploy)

        return trialsetlist

    def toJSON(self):
        """
        Export the entity to a JSON-compatible dictionary.

        Returns
        -------
        dict
            A dictionary with ``name``, ``entityType``, and ``trialProperties`` keys.
        """
        ret = dict()
        ret['name'] = self.name
        ret['entityType'] = self.entityType.name
        ret['trialProperties'] = self.allPropertiesList
        return ret

    def __str__(self):
        """Return the full JSON string representation of the entity."""
        return json.dumps(self.toJSON())

    def __repr__(self):
        """Return a compact JSON string with name and properties."""
        ret = dict(name=self.name, properties=self.propertiesList)
        return json.dumps(ret)


    @property
    def allPropertiesTable(self):
        """
        All properties (constant + trial-specific) as a combined DataFrame.

        Joins trial-specific properties with constant properties and
        forward-fills missing values.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per trial, containing all properties.
        """
        constantProperties = self.propertiesTable
        with pandas.option_context('future.no_silent_downcasting', True):
            ret = self.allTrialPropertiesTable.join(constantProperties).ffill().infer_objects(copy=False)
        return ret

    @property
    def propertiesTable(self):
        """
        The entity's constant properties as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``name``, ``value``, and ``scope``.
        """
        return pandas.DataFrame(self.propertiesList)

    @property
    def allTrialPropertiesTable(self):
        """
        Trial-specific properties across all trial sets and trials.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per trial, containing the entity's
            trial-specific properties plus ``trialSetName`` and ``trialName``
            columns.
        """
        retList = []
        for trialsSetsName in self.experiment.trialSet.keys():
            for trialName in self.experiment.trialSet[trialsSetsName].keys():
                trProp = self.trialProperties(trialsSetsName, trialName)
                trProp['trialSetName'] =trialsSetsName
                trProp['trialName'] = trialName
                retList.append(trProp)

        return pandas.DataFrame(retList)


    @property
    def allTrialProperties(self):
        """
        Trial-specific properties organized as a nested dictionary.

        Returns
        -------
        dict[str, dict[str, dict]]
            A nested dictionary: ``{trialSetName: {trialName: properties}}``.
        """
        trialsetdict = dict()

        for trialsSetsName in self.experiment.trialSet.keys():
            trialsetdict[trialsSetsName] = dict()
            for trialName in self.experiment.trialSet[trialsSetsName].keys():
                trialsetdict[trialsSetsName][trialName] = self.trialProperties(trialsSetsName, trialName)

        return trialsetdict

    def trialProperties(self, trialSetName, trialName):
        """
        Get this entity's properties for a specific trial.

        Parameters
        ----------
        trialSetName : str
            The name of the trial set (e.g., ``"design"``).
        trialName : str
            The name of the trial.

        Returns
        -------
        dict
            A dictionary of property names to values for this entity
            in the specified trial. Returns an empty dict if the entity
            is not part of the trial.
        """
        properties = self.experiment.trialSet[trialSetName][trialName].entities
        ret = properties.get(self.name, dict())
        return ret

    #################################### Deprecated entity interface ##################################

    def trial(self, trialSet, trialName, state):
        """
        Get trial properties by state.

        .. deprecated::
            Use :meth:`trialProperties` instead.

        Parameters
        ----------
        trialSet : str
            The trial set name.
        trialName : str
            The trial name.
        state : str
            ``'design'`` or ``'deploy'``.

        Returns
        -------
        dict
            The entity's properties for the specified trial and state.
        """
        warnings.warn("trial is deprecated. Use trialProperties", DeprecationWarning, stacklevel=2)
        return getattr(self, f"trial{state.title()}")(trialSet, trialName)

    @property
    def designProperties(self):
        """
        .. deprecated::
            Use :attr:`allTrialProperties` instead.
        """
        warnings.warn("designProperties is deprecated. Use allTriealProperties", DeprecationWarning, stacklevel=2)
        return self.propertiesList

    @property
    def deployProperties(self):
        """
        .. deprecated::
            Use :attr:`allTrialProperties` instead.
        """
        warnings.warn("deployProperties is deprecated. Use allTriealProperties", DeprecationWarning, stacklevel=2)
        return self.propertiesList

    def trialDesign(self, trialSet, trialName):
        """
        .. deprecated::
            Use :meth:`trialProperties` instead.
        """
        warnings.warn("trialDesign is deprecated. Use trialProperties", DeprecationWarning, stacklevel=2)
        return self.trialDeploy(trialSet, trialName)

    def trialDeploy(self, trialSet, trialName):
        """
        .. deprecated::
            Use :meth:`trialProperties` instead.
        """
        warnings.warn("trialDeploy is deprecated. Use trialProperties", DeprecationWarning, stacklevel=2)
        properties = self.experiment.trialSet[trialSet][trialName].deployEntities
        ret = properties.get(self.name, dict())

        return ret


####################################################################################################
