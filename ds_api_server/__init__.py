
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

from fastapi import FastAPI,Request,Response,status,UploadFile,File,Form,Body,Depends,HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import traceback
import time
import os
import io
import numpy as np
from starlette.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
# import newrelic.agent

AUTH_USER=os.environ.get("USER","demo")
AUTH_PASSW=os.environ.get("PASSW","demo")

app = FastAPI(title="Data Science api server",version="1.0.0")
security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, AUTH_USER)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSW)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# from . import rpc
from . import controllers
