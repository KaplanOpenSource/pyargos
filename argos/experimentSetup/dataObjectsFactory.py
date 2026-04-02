"""
Factory for loading experiments from local files.

Provides ``fileExperimentFactory`` which loads experiments from local
files (JSON or ZIP) and returns ``Experiment`` (or ``ExperimentZipFile``)
objects.
"""

import glob
from argos.experimentSetup.dataObjects import Experiment, ExperimentZipFile
import os
from argos.utils.jsonutils import loadJSON
from argos.utils.logging import get_logger as argos_get_logger

class fileExperimentFactory:
    """
    Factory that loads experiment data from the local filesystem.

    Scans the experiment directory's ``runtimeExperimentData/`` folder for
    either a ``.zip`` file or an ``experiment.json`` file, and returns the
    appropriate Experiment object.

    Parameters
    ----------
    experimentPath : str, optional
        Path to the experiment root directory. Defaults to the current
        working directory if not specified.

    Examples
    --------
    >>> factory = fileExperimentFactory("/path/to/experiment")
    >>> experiment = factory.getExperiment()
    >>> print(experiment.name)
    """

    basePath = None

    def __init__(self, experimentPath=None):
        """
        Initialize the file experiment factory.

        Parameters
        ----------
        experimentPath : str, optional
            Path to the experiment root directory. If None, uses the
            current working directory.
        """
        self.basePath = os.getcwd() if experimentPath is None else experimentPath
        self.logger = argos_get_logger(self)

    def getExperiment(self):
        """
        Load the experiment from the filesystem.

        Searches ``[experimentPath]/runtimeExperimentData`` for experiment data.
        If a ``.zip`` file is found, returns an ``ExperimentZipFile``.
        Otherwise, attempts to load ``experiment.json`` and returns an ``Experiment``.

        Returns
        -------
        Experiment or ExperimentZipFile
            The loaded experiment object.

        Raises
        ------
        ValueError
            If neither a ZIP file nor ``experiment.json`` can be found in the
            experiment data directory.
        """
        experimentAbsPath = os.path.abspath(os.path.join(self.basePath,"runtimeExperimentData"))

        # Scan the directory to check if there is a .zip file.
        zipped = False

        zipfileList = [fle for fle in glob.glob(os.path.join(experimentAbsPath,"*.zip"))]
        if len(zipfileList) == 0:
            self.logger.info(f"Cannot find zip files in the {experimentAbsPath}, trying to load the experiment.json file")
            datafile = os.path.join(experimentAbsPath, "experiment.json")
            if not os.path.isfile(datafile):
                experimentDict = loadJSON(datafile)
            else:
                err = f"cannot find experiment.json in the directory {os.path.join(experimentAbsPath)}"
                self.logger.error(err)
                raise ValueError(err)
        else:
            zipped = True
            self.logger.info(f"Found zip files: {zipfileList}. Taking the first: {zipfileList[0]}")
            experimentDict = zipfileList[0]

        if zipped:
            ret =  ExperimentZipFile(setupFileOrData=experimentDict)
        else:
            ret =  Experiment(setupFileOrData=experimentDict)

        self.logger.info("------------- End ----------")
        return ret

    def __getitem__(self, item):
        """
        Load an experiment by path using dictionary-style access.

        Parameters
        ----------
        item : str
            The experiment path.

        Returns
        -------
        Experiment or ExperimentZipFile
            The loaded experiment object.
        """
        return self.getExperiment(experimentPath=item)
