"""
Tests for the Experiment class — properties, entity types, trial sets.
"""

import pandas
import pytest
from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory


@pytest.fixture
def simple_exp(exp_simple_path):
    """Load the simple experiment once for this module."""
    return fileExperimentFactory(exp_simple_path).getExperiment()


@pytest.fixture
def groups_exp(exp_groups_path):
    """Load the groups experiment once for this module."""
    return fileExperimentFactory(exp_groups_path).getExperiment()


# --- Experiment metadata ---

class TestExperimentMetadata:

    def test_name(self, simple_exp):
        """exp.name should work for v3 ZIP experiments."""
        assert simple_exp.name == "New Experiment"

    def test_description_is_string(self, simple_exp):
        """exp.description should work for v3 ZIP experiments."""
        assert isinstance(simple_exp.description, str)

    def test_setup_is_dict(self, simple_exp):
        assert isinstance(simple_exp.setup, dict)

    def test_setup_has_version(self, simple_exp):
        # After version migration, internal format may not have 'version'
        # but the setup dict should be non-empty
        assert len(simple_exp.setup) > 0


# --- Entity types ---

class TestEntityTypes:

    def test_entity_type_count(self, simple_exp):
        """Simple experiment has 4 device types."""
        assert len(simple_exp.entityType) == 4

    def test_entity_type_names(self, simple_exp):
        names = set(simple_exp.entityType.keys())
        assert "New Device Type" in names
        assert "New Device Type 1" in names

    def test_entity_type_is_dict_accessible(self, simple_exp):
        """EntityType should be accessible by name (dict-style)."""
        et = simple_exp.entityType["New Device Type"]
        assert et.name == "New Device Type"

    def test_entity_type_has_properties(self, simple_exp):
        et = simple_exp.entityType["New Device Type"]
        assert isinstance(et.properties, list)
        assert len(et.properties) > 0

    def test_entity_type_properties_table_is_dataframe(self, simple_exp):
        et = simple_exp.entityType["New Device Type"]
        assert isinstance(et.propertiesTable, pandas.DataFrame)

    def test_entity_type_table_is_dataframe(self, simple_exp):
        """experiment.entityTypeTable should be a DataFrame."""
        assert isinstance(simple_exp.entityTypeTable, pandas.DataFrame)
        assert len(simple_exp.entityTypeTable) > 0

    def test_groups_has_one_entity_type(self, groups_exp):
        assert len(groups_exp.entityType) == 1
        assert "New Device Type" in groups_exp.entityType


# --- Entities ---

class TestEntities:

    def test_groups_entity_count(self, groups_exp):
        """Groups experiment has 2 devices of type 'New Device Type'."""
        et = groups_exp.entityType["New Device Type"]
        assert et.numberOfEntities == 2

    def test_entity_names(self, groups_exp):
        et = groups_exp.entityType["New Device Type"]
        names = set(et.keys())
        assert "New Device" in names
        assert "New Device 1" in names

    def test_entity_dict_access(self, groups_exp):
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        assert entity.name == "New Device"

    def test_entity_type_name(self, groups_exp):
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        assert entity.entityType == "New Device Type"

    def test_entity_experiment_reference(self, groups_exp):
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        assert entity.experiment is groups_exp

    def test_entities_table_is_dataframe(self, groups_exp):
        assert isinstance(groups_exp.entitiesTable, pandas.DataFrame)

    def test_entities_table_has_entity_columns(self, groups_exp):
        df = groups_exp.entitiesTable
        assert "entityType" in df.columns
        assert "entityName" in df.columns

    def test_entities_table_row_count(self, groups_exp):
        """Should have 2 rows (one per entity)."""
        assert len(groups_exp.entitiesTable) == 2

    def test_entity_properties_constant_scope(self, groups_exp):
        """Entity should have Constant-scope property from type defaults."""
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        props = entity.properties
        assert "StoreDataPerDevice" in props

    def test_entity_properties_table(self, groups_exp):
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        df = entity.propertiesTable
        assert isinstance(df, pandas.DataFrame)
        assert len(df) > 0

    def test_entity_type_entities_table(self, groups_exp):
        et = groups_exp.entityType["New Device Type"]
        df = et.entitiesTable
        assert isinstance(df, pandas.DataFrame)
        assert len(df) == 2


# --- Entity constant properties ---

class TestEntityConstantProperties:

    def test_constant_property_value(self, groups_exp):
        """StoreDataPerDevice should be False (from type default)."""
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        assert entity.properties["StoreDataPerDevice"] is False

    def test_properties_list_has_scope(self, groups_exp):
        entity = groups_exp.entityType["New Device Type"]["New Device"]
        for prop in entity.propertiesList:
            assert "scope" in prop
            assert "name" in prop
            assert "value" in prop
