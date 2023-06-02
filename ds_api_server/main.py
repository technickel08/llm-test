import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

import importlib
import os
import queue
import json
import pprint
import sys
from starlette.responses import RedirectResponse
import uvicorn

from ds_api_server import app
# Get the kafka address
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8080, workers=1024)


    @app.get("/", include_in_schema=False)
    def docs_redirect():
        return RedirectResponse(f"/redoc")
