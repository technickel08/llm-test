import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

# import gradio as gr
import openai
import os
from functools import partial
import time
import os
# import openai
import whisper
from gtts import gTTS
# import chat_agent
from langchain.schema import (
    HumanMessage
)
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.llms import OpenAI
# os.environ["OPENAI_API_KEY"] = "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"
    

def text2audio(text):
    logger.info("audio 2 text initialized")
    myobj = gTTS(text=text, lang='en', slow=False) 
    myobj.save("audio_out.wav")
    logger.info("audio 2 text complete")
    mp3_fp = BytesIO()
    myobj.write_to_fp(mp3_fp)
    logger.info("audio 2 text file to bytes compelete")
    return "audio_out.wav"




class ResponseHeaderV1(BaseModel):
    # created_at : str
    text: str


class ResponseHeaderV2(BaseModel):
    # created_at : str
    text: str
    language_code: str
    voice_gender : str