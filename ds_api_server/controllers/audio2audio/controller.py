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
from .utils import ResponseHeaderV1,ResponseHeaderV2
import traceback
from fastapi import FastAPI,Request,Response,status,UploadFile,File,Form,Body,HTTPException,Depends
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from langchain.llms import OpenAI
import openai
from .utils import ChatBot
# import redis
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from fastapi.responses import FileResponse
from google.cloud import texttospeech
from ds_api_server import app,connections,get_current_username
from google.cloud import texttospeech


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


CHATBOT = ChatBot()

@app.post('/audio_to_audio', status_code=200)
def audio2text_v2_api_call(
    user_id : str,
    selected_model : str =  "gpt-3.5-turbo",
    language_code :str = "hi-IN",
    voice_gender : str ="FEMALE",
    audio : UploadFile = File(None,description="Upload audio file"),
                        language:Optional[str]="en",
                        context_enable :bool = False,
                        openai_turbo_gpt :bool = False,
                        hugging_face_gpt:bool = False) -> JSONResponse:
    
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
            audio= audio.file
            logger.info("file read")
        else:
            logger.info("no audio")
        logger.info("processing file")
        resp = utils.audio2text_v2(audio,language=language)
        logger.info("stt received text - {}".format(str(resp)))
        logger.info("file processed")
        # print(resp,type(resp))
        out = CHATBOT.conversation(resp,selected_model,user_id,context_enable,openai_turbo_gpt,hugging_face_gpt)
        logger.info("llm received text - {}".format(str(out)))
        header = {"text":out,
                  "language_code": language_code,
                  "voice_gender": voice_gender}
        result = text2audio_api_call_v2(header)
        # out["result"] = {"text":resp}
        # out["resp_time"] = round(time.time()-t1,2)
        logger.info("complete")
        return FileResponse(result)
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




def text2audio_api_call_v2(
    text : ResponseHeaderV2,
    # audio : UploadFile = File(None,description="Upload audio file")
    ) -> StreamingResponse:
    """
    TTS : Text to speech conversion
    Args:
        text (ResponseHeaderV1): text to convert

    Returns:
        StreamingResponse: audio streaming output file
    """

    t1 = time.time()
    client = texttospeech.TextToSpeechClient()
    # text = text.dict()
    voice_gender=text["voice_gender"]
    language_code=text["language_code"]
    text=text["text"]
    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    if voice_gender == "FEMALE":
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name="en-IN-Standard-D",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
    else:
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name="en-IN-Standard-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )
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
    return "output.mp3"