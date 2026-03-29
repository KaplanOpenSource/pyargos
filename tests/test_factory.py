"""
Tests for fileExperimentFactory — loading experiments from different sources.
"""

import pytest
from argos.experimentSetup.dataObjectsFactory import fileExperimentFactory
from argos.experimentSetup.dataObjects import Experiment, ExperimentZipFile


class TestFileExperimentFactorySimple:
    """Tests using the simple experiment (v3.0.0, no devices on trial)."""

    def test_factory_returns_experiment(self, exp_simple_path):
        """Factory should return an ExperimentZipFile for a ZIP-based experiment."""
        factory = fileExperimentFactory(exp_simple_path)
        exp = factory.getExperiment()
        assert isinstance(exp, ExperimentZipFile)

    def test_factory_loads_correct_name(self, exp_simple_path):
        """Loaded experiment should have the correct name from data.json."""
        exp = fileExperimentFactory(exp_simple_path).getExperiment()
        assert exp.name == "New Experiment"

    def test_factory_default_path_is_cwd(self):
        """Factory with no path should default to current working directory."""
        factory = fileExperimentFactory()
        assert factory.basePath is not None


class TestFileExperimentFactoryGroups:
    """Tests using the groups experiment (v3.0.0, containment)."""

    def test_factory_loads_groups_experiment(self, exp_groups_path):
        """Factory should load the groups experiment successfully."""
        exp = fileExperimentFactory(exp_groups_path).getExperiment()
        assert exp.name == "exp groups"

    def test_factory_loads_noattr_experiment(self, exp_groups_noattr_path):
        """Factory should load the no-attributes experiment successfully."""
        exp = fileExperimentFactory(exp_groups_noattr_path).getExperiment()
        assert exp.name == "exp groups"


class TestFileExperimentFactoryInvalidPath:
    """Tests for error handling with invalid paths."""

    def test_factory_invalid_path_raises(self, tmp_path):
        """Factory with a path that has no experiment data should raise ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        (empty_dir / "runtimeExperimentData").mkdir()
        with pytest.raises((ValueError, FileNotFoundError)):
            fileExperimentFactory(str(empty_dir)).getExperiment()
