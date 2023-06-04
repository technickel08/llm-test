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
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from fastapi import FastAPI,Response,status,Request,Depends
from typing import Optional
from . import utils
from .utils import ResponseHeaderV1,ResponseHeaderV2
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



# @app.post('/text2audio', status_code=200)
# def text2audio_api_call(
#     text : ResponseHeaderV1,
#     # audio : UploadFile = File(None,description="Upload audio file")
#     ) -> StreamingResponse:
#     """
#     TTS : Text to speech conversion
#     Args:
#         text (ResponseHeaderV1): text to convert

#     Returns:
#         StreamingResponse: audio streaming output file
#     """

#     t1 = time.time()
#     text = text.dict()
#     text = text["text"]
#     print(text)
#     out = {
#             "status":None,
#             "message":None,
#             "results":{}
#         }
#     try:
#         # audio=audio.read()
#         if text is not None:  
#             logger.info("reading file")
#             resp = utils.text2audio(text)
#             logger.info("file read")
#         else:
#             logger.info("no audio")
#         # logger.info("processing file")
#         # resp = utils.audio2text_v2(audio)
#         # logger.info("file processed")
#         print(type(resp))
#         out["result"] = {"file":resp}
#         out["resp_time"] = round(time.time()-t1,2)
#         logger.info("complete")
#         return FileResponse(resp)
#     # StreamingResponse(
#     #         content=resp,
#     #         status_code=status.HTTP_200_OK,
#     #         media_type="audio/wav",
#     #     )
#     except Exception as e:
#         logger.error("some exception occurred-{}".format(str(e)))
#         logger.error(traceback.format_exc())
#         out={}
#         out["status"]='FAILED'
#         out["message"]=str(traceback.format_exc())
#         return JSONResponse(status_code=400,content=out)



@app.post('/text2audio', status_code=200)
def text2audio_api_call_v2(
    text : ResponseHeaderV2

    # audio : UploadFile = File(None,description="Upload audio file")
    ) -> StreamingResponse:
    """
    TTS : Text to speech conversion
    Args:
        text (ResponseHeaderV1): text to convert
        languages-codes-names-gender
        English (India)	Standard	en-IN	en-IN-Standard-A	FEMALE	
        English (India)	Standard	en-IN	en-IN-Standard-B	MALE
        English (India)	WaveNet	en-IN	en-IN-Wavenet-A	FEMALE	
        English (India)	WaveNet	en-IN	en-IN-Wavenet-B	MALE
        Hindi (India)	Standard	hi-IN	hi-IN-Standard-A	FEMALE	
        Hindi (India)	Standard	hi-IN	hi-IN-Standard-B	MALE
        Hindi (India)	WaveNet	hi-IN	hi-IN-Wavenet-A	FEMALE	
        Hindi (India)	WaveNet	hi-IN	hi-IN-Wavenet-B	MALE

    Returns:
        StreamingResponse: audio streaming output file
    """

    t1 = time.time()
    try:
        client = texttospeech.TextToSpeechClient()
        text = text.dict()
        voice_gender=text["voice_gender"]
        language_code=text["voice_code"]
        voice_name= text["voice_name"]
        text=text["text"]
        input_text = texttospeech.SynthesisInput(text=text)

        # Note: the voice can also be specified by name.
        # Names of voices can be retrieved with client.list_voices().
        # if voice_gender == "FEMALE":
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=voice_gender#texttospeech.SsmlVoiceGender.FEMALE,
        )
        # else:
        #     voice = texttospeech.VoiceSelectionParams(
        #         language_code=language_code,
        #         name="en-IN-Standard-B",
        #         ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        #     )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        # The response's audio_content is binary.
        with open("output.mp3", "wb") as out:
            out.write(response.audio_content)
            print('Audio content written to file "output.mp3"')
            
        return FileResponse("output.mp3")
    
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)
