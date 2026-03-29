"""
Tests for version migration — loading v2.0.0 and v3.0.0 ZIP experiments.
"""

import pytest
import pandas
from argos.experimentSetup.dataObjects import ExperimentZipFile


class TestVersion3:
    """Tests for v3.0.0 format (current, exp_simple and exp_groups)."""

    def test_loads_v3_simple(self, exp_simple_path):
        """v3.0.0 simple experiment should load without errors."""
        from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory
        exp = fileExperimentFactory(exp_simple_path).getExperiment()
        assert exp.name == "New Experiment"

    def test_loads_v3_groups(self, exp_groups_path):
        """v3.0.0 groups experiment should load without errors."""
        from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory
        exp = fileExperimentFactory(exp_groups_path).getExperiment()
        assert exp.name == "exp groups"

    def test_v3_has_entity_types(self, exp_simple_path):
        """v3.0.0 deviceTypes should be normalized to entityType."""
        from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory
        exp = fileExperimentFactory(exp_simple_path).getExperiment()
        assert len(exp.entityType) == 4

    def test_v3_has_trial_sets(self, exp_simple_path):
        """v3.0.0 trialTypes should be normalized to trialSet."""
        from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory
        exp = fileExperimentFactory(exp_simple_path).getExperiment()
        assert len(exp.trialSet) == 2


class TestVersion2:
    """Tests for v2.0.0 format (Raptor2023).

    Note: v2.0.0 migration has a known issue where trialSet metadata
    uses 'properties' instead of 'attributeTypes', causing TrialSet and
    Trial initialization to fail with KeyError. These tests are marked
    xfail until the migration is fixed.
    """

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_loads_v2_raptor(self, raptor_zip_path):
        """v2.0.0 experiment should load and migrate without errors."""
        exp = ExperimentZipFile(raptor_zip_path)
        assert isinstance(exp, ExperimentZipFile)

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_has_entity_types(self, raptor_zip_path):
        """v2.0.0 entities should be nested into entityTypes after migration."""
        exp = ExperimentZipFile(raptor_zip_path)
        assert len(exp.entityType) == 8

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_has_trial_sets(self, raptor_zip_path):
        """v2.0.0 trials should be nested into trialSets after migration."""
        exp = ExperimentZipFile(raptor_zip_path)
        assert len(exp.trialSet) == 1

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_entity_type_has_entities(self, raptor_zip_path):
        """After migration, entity types should contain their entities."""
        exp = ExperimentZipFile(raptor_zip_path)
        first_type = list(exp.entityType.values())[0]
        assert first_type.numberOfEntities > 0

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_trial_set_has_trials(self, raptor_zip_path):
        """After migration, trial sets should contain their trials."""
        exp = ExperimentZipFile(raptor_zip_path)
        first_ts = list(exp.trialSet.values())[0]
        assert len(first_ts) > 0

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_entities_table(self, raptor_zip_path):
        """entitiesTable should work after v2 migration."""
        exp = ExperimentZipFile(raptor_zip_path)
        df = exp.entitiesTable
        assert isinstance(df, pandas.DataFrame)
        assert len(df) > 0

    @pytest.mark.xfail(reason="v2.0.0 migration: trialSet 'properties' not renamed to 'attributeTypes'", strict=True)
    def test_v2_entity_type_name(self, raptor_zip_path):
        """First entity type should be 'sonic'."""
        exp = ExperimentZipFile(raptor_zip_path)
        assert "sonic" in exp.entityType
