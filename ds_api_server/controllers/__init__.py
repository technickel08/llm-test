import os
import json
import importlib
from service_folder import SERVICE_FOLDER

sf = os.environ.get("SERVICE_FOLDER",None)
if sf:
    SERVICE_FOLDER=json.loads(sf)


for key,values in SERVICE_FOLDER.items():
    if values is True:
        importlib.import_module('controllers.{}'.format(key))