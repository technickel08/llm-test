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


def audio2text(file)->str:
    """
    Using OpenAI Whisper API to convert audio 2 text
    Args:
        file (_type_): Audio file, File Formats supported : mp3,wav

    Returns:
        str: Converted text
    """    
    logger.info("Whisper API call initialised")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # audio_file= open(file, "rb")
    logger.info("audo file loaded")
    t1 = time.time()
    result = openai.Audio.transcribe("whisper-1",file)
    logger.info("Whisper API inference compelete - {}s".format(round(time.time()-t1)))
    logger.info("API call Function executed")
    return result.text


def audio2text_v2(file,language="en"):
    """
    Using Google TTS Speech Recognition API to convert audio 2 text
    Args:
        file (_type_): Audio file, File Formats supported : mp3,wav

    Returns:
        str: Converted text
    """
    try:
        logger.info("Speech Recognition API call initialised")
        r = sr.Recognizer()
        logger.info("Speech Recognition object initialised")
        try:
            sound = AudioSegment.from_mp3(file)
        except: 
            sound = AudioSegment.from_wav(file)
        sound.export("input_audio.wav", format="wav")
        audio_file = sr.AudioFile("input_audio.wav")
        logger.info("audio file saved")
        with audio_file as source:
            audio = r.record(source)
        logger.info("audo file loaded")
        t1 = time.time()
        result = r.recognize_google(audio,language=language)
        logger.info("speech recognition API inference compelete - {}s".format(round(time.time()-t1)))
        logger.info("API call Function executed")
        return str(result)
    except Exception as e:
        logger.error("some exception occured - {}".format(str(e)))
        raise