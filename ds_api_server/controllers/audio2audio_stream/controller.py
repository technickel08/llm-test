from datetime import datetime
import logging
import datetime as dt
from tkinter import EXCEPTION
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)
import base64
import time
import os
from starlette.responses import RedirectResponse
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
from . import utils_search
from .utils import RequestHeaderV1,RequestHeaderV2,ChatBot
import traceback
from fastapi import FastAPI,Request,Response,status,UploadFile,File,Form,Body,HTTPException,Depends
from fastapi.responses import FileResponse
# from starlette.responses import StreamingResponse
import json
from fastapi.responses import StreamingResponse
from langchain.llms import OpenAI
import openai
# import redis
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from fastapi.responses import FileResponse
from google.cloud import texttospeech
from ds_api_server import app,connections,get_current_username
import asyncio
from sse_starlette.sse import EventSourceResponse
from queue import Queue
from threading import Thread
from google.cloud import texttospeech
from controllers.audio2text.utils import audio2text_v2
import uuid

# app = FastAPI()
CHATBOT = ChatBot()
CHATBOT_SEARCH = utils_search.ChatBot()

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



STREAM_DELAY = 0.00001  # second
RETRY_TIMEOUT = 15000  # milisecond


class TTS():
    def __init__(self,voice_code,voice_gender,voice_name):
        self.voice_gender=voice_gender
        self.language_code=voice_code
        self.voice_name= voice_name
    def run(self,text : str):
        client = texttospeech.TextToSpeechClient()
        # text = text.dict()
        # voice_gender=text["voice_gender"]
        # language_code=text["voice_code"]
        # voice_name= text["voice_name"]
        # text=text["text"]
        input_text = texttospeech.SynthesisInput(text=text)

        # Note: the voice can also be specified by name.
        # Names of voices can be retrieved with client.list_voices().
        # if voice_gender == "FEMALE":
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
            ssml_gender=self.voice_gender#texttospeech.SsmlVoiceGender.FEMALE,
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
        return response.audio_content
#base64.b64decode(response.audio_content)

@app.post('/audio2audio_stream')
async def message_stream(
                request: Request,
                user_id:int,
                text : Optional[str]=None,
                selected_model :str = "gpt-3.5-turbo",
                context_enable :bool = False ,
                internet_enable :bool = False ,
                voice_code :str = "en-IN",
                voice_gender :str = "FEMALE",
                voice_name :str = "en-IN-Standard-A",
                tts_lang : str = "en",
                audio : Optional[UploadFile] = File(None,description="Upload audio file"),
                session_id : Optional[str] = uuid.uuid1(),
                username: str = Depends(get_current_username)
                ):
    q = Queue()
    t1 = time.time()
    if text is None and audio is not None:
        logger.info("---- No text found ----")
        logger.info("---- audio found ----")
        user_input = audio2text_v2(audio.file,tts_lang)
    else:
        logger.info("---- No audio found ----")
        logger.info("---- Text found ----")
        user_input = str(text)
    logger.info("audio to text output - ".format(user_input))
    if internet_enable:
        thread = Thread(target=CHATBOT_SEARCH.conversation(user_input,selected_model,user_id,context_enable,q,str(session_id)).run, kwargs={"input": user_input})
    else:
        thread = Thread(target=CHATBOT.conversation(user_input,selected_model,user_id,context_enable,q,str(session_id)).run, kwargs={"text": user_input})
    tts = TTS(voice_code,voice_gender,voice_name)
    thread.start()
    async def event_generator():
            uid = uuid.uuid1()
            t1 = time.time()
            word = ""
            final_out = ""
            print(time.time()-t1,"*"*10,"async generator")
            internet_out = False
            while True:
                if q.qsize()>0:
                    token = q.get()
                    if token in ["Final","Answer"]:
                        internet_out = True
                    # print("token nikala")
                    if internet_enable is False or internet_out is True:
                        print(token)
                        final_out += token
                        if token != "DONE":
                            word+= token
                        if token in [".",","," ","।","?"]:
                        # print(time.time()-t1)
                            print(word,time.time()-t1)
                            yield {
                                        "event": str(word),
                                        "id": uid,
                                        "data": tts.run(word)
                                }
                            print(time.time()-t1)
                            word = ""
                        if token == "DONE":
                            yield {
                                        "event": str(word),
                                        "id": uid,
                                        "data": tts.run(word)
                                }
                            break
                    t1 = time.time()
        # while True:
            # If client closes connection, stop sending events
                if await request.is_disconnected():
                    print("connection terminated")
                    break

                # Checks for new messages and return them to client if any
            CHATBOT.update_context(user_input,final_out,user_id,str(session_id))
            await asyncio.sleep(STREAM_DELAY)
    #thread.join()
    return EventSourceResponse(event_generator())




async def generate(answer,text):
    for line in answer.run(text):
        # print(line)
        yield line


# @app.post('/conversation_stream', status_code=200)
# async def conversation(user_id:int,text : ResponseHeaderV1,
#                  selected_model :str = "gpt-3.5-turbo",
#                  context_enable :bool = False ) -> dict:
#     """
#     This function records user reponse and search previous session context in the memory
#     Bot is designed to process responses empatheti

#     Args:
#         user_id (int): user_id to restore session context
#         text (ResponseHeaderV1): user input query
#         selected_model (str, optional): Huggingface/OpenAI model selection. Defaults to "gpt-3.5-turbo".

#     Returns:
#         dict: { result : LLM output,
#                 "resp_time": time taken to record the reponse}
#     """    
#     try:
#         logger.info("llm conversation initiated")
#         t1 = time.time()
#         user_input = text.dict()["text"]
#         answer = CHATBOT.conversation(text,selected_model,user_id,context_enable)
#         # resp = answer.run(text)
#         # context = redis_db.get(user_id,"")
#         # chat = ChatOpenAI(temperature=0,model_name=selected_model)
#         # template = """You are a companion, it is your job to talk to me with empathy.
#         # Question: {text}
#         # Answer:
#         # """
#         # # redis_connect.get(user_id)
#         # # redis_connect.set('some_key', context)
#         # prompt_template = PromptTemplate(input_variables=["text"], template=template)
#         # answer_chain = LLMChain(llm=chat, prompt=prompt_template)
#         # answer = answer_chain.run(user_input)
#         # memory = ConversationBufferMemory(return_messages=True)
#         # memory.chat_memory.add_user_message(user_input)
#         # memory.chat_memory.add_ai_message(answer)
#         # history = memory.load_memory_variables({})
#         # context["history"].append(history)
#         # redis_db.set(user_id, context)
#         logger.info("llm conversation compelete")
#         return StreamingResponse(generate(answer,text), media_type="text/event-stream")
#     except Exception as e:
#         logger.error("some exception occurred-{}".format(str(e)))
#         logger.error(traceback.format_exc())
#         out={}
#         out["status"]='FAILED'
#         out["message"]=str(traceback.format_exc())
#         return JSONResponse(status_code=400,content=out)


# Default route to documentation
@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"/redoc")