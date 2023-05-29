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
import utils
from utils import ResponseHeaderV1,ResponseHeaderV2
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


app = FastAPI()

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

@app.post('/audio_to_text', status_code=200)
def audio2text_v1_api_call(audio : UploadFile = File(None,description="Upload audio file"),
                        language:Optional[str]=None):
    """
    STT :  Speech to Text conversion
    Args:
        text (ResponseHeaderV1): Audio to convert

    Returns:
        JSONResponse: Converted Text and detected Language
    """    
    out = {
            "status":None,
            "message":None,
            "results":{}
        }
    try:
        resp = utils.audio2text(audio.file)
        out["result"] = {"text":resp.text}
        return out
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)


@app.post('/audio_to_text_v2', status_code=200)
def audio2text_v2_api_call(
    audio : UploadFile = File(None,description="Upload audio file"),
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
            audio= audio.file
            logger.info("file read")
        else:
            logger.info("no audio")
        logger.info("processing file")
        resp = utils.audio2text_v2(audio)
        logger.info("file processed")
        print(resp,type(resp))
        out["result"] = {"text":resp}
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

@app.post('/text2audio', status_code=200)
def text2audio_api_call(
    text : ResponseHeaderV1,
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
    text = text.dict()
    text = text["text"]
    print(text)
    out = {
            "status":None,
            "message":None,
            "results":{}
        }
    try:
        # audio=audio.read()
        if text is not None:  
            logger.info("reading file")
            resp = utils.text2audio(text)
            logger.info("file read")
        else:
            logger.info("no audio")
        # logger.info("processing file")
        # resp = utils.audio2text_v2(audio)
        # logger.info("file processed")
        print(type(resp))
        out["result"] = {"file":resp}
        out["resp_time"] = round(time.time()-t1,2)
        logger.info("complete")
        return FileResponse(resp)
    # StreamingResponse(
    #         content=resp,
    #         status_code=status.HTTP_200_OK,
    #         media_type="audio/wav",
    #     )
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)



@app.post('/text2audio_v2', status_code=200)
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
    try:
        client = texttospeech.TextToSpeechClient()
        text = text.dict()
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
        return FileResponse("output.mp3")
    # StreamingResponse(
    #         content=resp,
    #         status_code=status.HTTP_200_OK,
    #         media_type="audio/wav",
    #     )
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)


@app.post('/conversation', status_code=200)
def conversation(user_id:int,text : ResponseHeaderV1,
                 selected_model :str = "gpt-3.5-turbo") -> dict:
    """
    This function records user reponse and search previous session context in the memory
    Bot is designed to process responses empatheti

    Args:
        user_id (int): user_id to restore session context
        text (ResponseHeaderV1): user input query
        selected_model (str, optional): Huggingface/OpenAI model selection. Defaults to "gpt-3.5-turbo".

    Returns:
        dict: { result : LLM output,
                "resp_time": time taken to record the reponse}
    """    
    try:
        t1 = time.time()
        user_input = text.dict()["text"]
        context = redis_db.get(user_id,"")
        chat = ChatOpenAI(temperature=0,model_name=selected_model)
        template = """You are a companion, it is your job to talk to me with empathy.
        Question: {text}
        Answer:
        """
        # redis_connect.get(user_id)
        # redis_connect.set('some_key', context)
        prompt_template = PromptTemplate(input_variables=["text"], template=template)
        answer_chain = LLMChain(llm=chat, prompt=prompt_template)
        answer = answer_chain.run(user_input)
        # memory = ConversationBufferMemory(return_messages=True)
        # memory.chat_memory.add_user_message(user_input)
        # memory.chat_memory.add_ai_message(answer)
        # history = memory.load_memory_variables({})
        # context["history"].append(history)
        # redis_db.set(user_id, context)
        return {"result":answer,"resp_time":time.time()-t1}
    except Exception as e:
        logger.error("some exception occurred-{}".format(str(e)))
        logger.error(traceback.format_exc())
        out={}
        out["status"]='FAILED'
        out["message"]=str(traceback.format_exc())
        return JSONResponse(status_code=400,content=out)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8080)

    #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