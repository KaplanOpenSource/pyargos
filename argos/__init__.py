__version__ = '1.3.0'
from .experimentSetup import FILE

from .manager import experimentManager

# Have some initial default logging configuration in case the user hasn't set any
from .utils.logging.helpers import initialize_logging
initialize_logging(disable_existing_loggers=False)


"""
Version 1.3.0
-------------

    * Added MkDocs documentation with GitHub Pages (User Guide + Developer Guide)
    * Added mkdocstrings autodoc for all modules (experimentSetup, manager, kafka, nodered, nosql, utils)
    * Added comprehensive numpy-style docstrings to all public classes, methods, and properties
    * Added Data Model Reference documenting ZIP file format, JSON schemas (v1/v2/v3), and domain concepts
    * Added experiment lifecycle documentation (planning, execution, post-processing)
    * Added architecture documentation: class dependencies, swimlane diagrams, and implementation notes for all modules
    * Pre-rendered all mermaid diagrams to SVG with render_diagrams.py (Docker-based)
    * Added Makefile with install, env, and docs targets
    * Added make env-persist to configure PYTHONPATH with shell profile prompt
    * Updated README and installation guide with correct Python version and make workflow
    * Fixed CI: removed requirements.txt from docs workflow (pillow build failure on Python 3.14)

Version 1.2.3
-------------

    * Added the thingsboard interface.
    * Remove depdnecey on the thingsboard interface library
    * fixed the case when maps does not appear.

Version 1.2.2
-------------

    Minor changes to support the old version of the configuration file. 

Version 1.2.1
-------------
    Fixed a bug reading the zipFile. However, this might break the old version. 
    We have to check it with one of the old experiments. 

Version 1.2.0
-------------

- Added a write to parquet and  append to parquet
- refactoring ths argos-experiment-manager to : 
    1. Build new experiment. 
    2. have kafaToParquet python utility. 
       the can either work independently by command or by message from a different topic. 

- adding logging utility. 


- Fixed the dataobject. Parses the type of the object
        Only parses text,number and location. 
        Should expand to datetime, bool and ect. 
        
- Fixed the loading trial: removes old attributes before loading.  
     
- Adding the argos-experiment-manager to setup, and load trials to thingsboard. 

- Fixing the dataObject with the new DB structure. 
   ** still did not add the contains property to the library.

Version 1.1.0
-------------

- Added factory to handle with JSON version 2.0.0 and all experiment data in zip file. 

Version 1.0.0
-------------

- Changes devices->entities

Version 0.4.0
-------------

- Added default assets. (windows and device groups).  
- Loading devices properties for the requested release. 


Version 0.2.0
-------------

* Refactoring the graphQL interface  
* Removed the report 

Version 0.1.0
-------------
* Added graphQL interface
* Started using kafka consumers(processors)

Version 0.0.2
-------------

* switched to the new swagger wrapping. 

getAttributes still doesnt work. 

Version 0.0.1
--------------

* Addition/removal of devices and assets.
* Updating attributes. 
* adding relations. 

TODO: We need to update the tb_api_client and then simplify some of the code 
and allow for further functionality. 

 



"""

# ---------------- to do -----------------------------
# from hera import METEOROLOGICAL, DISPERSION, ...
# deviceTypeToDocTypeDict = dict(Raw_Sonic=METEOROLOGICAL,
#                                NDIR=DISPERSION,
#                                .
#                                .
#                                .
#                                )
