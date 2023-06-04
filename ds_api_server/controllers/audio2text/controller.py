from datetime import datetime
import logging
import datetime as dt
from tkinter import EXCEPTION
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

import time
import os
import uvicorn
# from fastapi_redis_cache import FastApiRedisCache,cache_one_hour
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from fastapi import FastAPI,Response,status,Request,Depends
from typing import Optional
from . import utils
# from utils import ResponseHeaderV1,ResponseHeaderV2
import traceback
from fastapi import FastAPI,Request,Response,status,UploadFile,File,Form,Body,HTTPException,Depends
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from langchain.llms import OpenAI
import openai
# import redis
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from fastapi.responses import FileResponse
from google.cloud import texttospeech
from ds_api_server import app,connections,get_current_username
import io


# app = FastAPI()

## Initialised redis connection
# redis_connect = redis.StrictRedis(host='0.0.0.0', port=6379, db=0)
# try:
#     redisHost = '0.0.0.0'
#     redisPort = 6379
#     REDIS_URL = "redis://{}:{}".format(redisHost, redisPort)
#     redis_db = redis.Redis.from_url(REDIS_URL)
#     logger.debug("redis db loaded-{}".format(REDIS_URL))
# except Exception as e:
#     logger.error("redis db not loaded-{}".format(str(e)))
#     redis_db = None


redis_db = {123:"hello"}

# @app.post('/audio_to_text', status_code=200)
# def audio2text_v1_api_call(audio : UploadFile = File(None,description="Upload audio file"),
#                         language:Optional[str]=None):
#     """
#     STT :  Speech to Text conversion
#     Args:
#         text (ResponseHeaderV1): Audio to convert

#     Returns:
#         JSONResponse: Converted Text and detected Language
#     """    
#     out = {
#             "status":None,
#             "message":None,
#             "results":{}
#         }
#     try:
#         resp = utils.audio2text(audio.file)
#         out["result"] = {"text":resp.text}
#         return out
#     except Exception as e:
#         logger.error("some exception occurred-{}".format(str(e)))
#         logger.error(traceback.format_exc())
#         out={}
#         out["status"]='FAILED'
#         out["message"]=str(traceback.format_exc())
#         return JSONResponse(status_code=400,content=out)


@app.post('/audio_to_text', status_code=200)
async def audio2text_v2_api_call(
    audio : UploadFile = File(None,description="Upload audio file"),
    model_whisper :bool = False,
    model_gtts: bool=True,
    language:Optional[str]=None) -> JSONResponse:
    
    """
    STT :  Speech to Text conversion
    Args:
        text (ResponseHeaderV1): Audio to convert

    Returns:
        JSONResponse: Converted Text and detected Language
    """

    t1 = time.time()
    out = {
            "status":None,
            "message":None,
            "results":{}
        }
    try:
        # audio=audio.read()
        if audio is not None:  
            logger.info("reading file")
        else:
            logger.info("no audio")
        logger.info("processing file")
        if model_whisper==True:
            buf =audio.file.read()
            buffer = io.BytesIO(buf)
            buffer.name = audio.filename
            resp = utils.audio2text(buffer)
        else:
            resp = utils.audio2text_v2(audio.file,str(language))
        logger.info("file processed")
        print(resp,type(resp))
        out["result"] = {"text":resp,"language":str(language)}
        out["resp_time"] = round(time.time()-t1,2)
        logger.info("complete")
        return out
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)

# Default route to documentation
@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"/redoc")