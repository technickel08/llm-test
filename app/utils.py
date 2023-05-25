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
import chat_agent
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


def audio2text_v2(file):
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
        sound = AudioSegment.from_mp3(file)
        sound.export("input_audio.wav", format="wav")
        audio_file = sr.AudioFile("input_audio.wav")
        with audio_file as source:
            audio = r.record(source)
        logger.info("audo file loaded")
        t1 = time.time()
        result = r.recognize_google(audio)
        logger.info("speech recognition API inference compelete - {}s".format(round(time.time()-t1)))
        logger.info("API call Function executed")
        return str(result)
    except Exception as e:
        logger.error("some exception occured - {}".format(str(e)))
        return None
    

def text2audio(text):
    logger.info("audio 2 text initialized")
    myobj = gTTS(text=text, lang='en', slow=False) 
    logger.info("audio 2 text complete")
    mp3_fp = BytesIO()
    myobj.write_to_fp(mp3_fp)
    logger.info("audio 2 text file to bytes compelete")
    return mp3_fp
    
class ChatBot:
    def __init__(self, memory, agent_chain):
        self.memory = memory
        self.agent = agent_chain

def create_chatbot(model_name, seed_memory=None):
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for all question that asks about live events",
        ),
        Tool(
            name="Topic Search",
            func=search.run,
            description="useful for all question that are related to a particular topic, product, concept, or service",
        )
    ]
    memory = seed_memory if seed_memory is not None else ConversationBufferMemory(memory_key="chat_history")
    chat = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
    # chain = LLMChain(llm=chat)
    agent_chain = initialize_agent(tools, chat, agent="conversational-react-description", verbose=True, memory=memory)

    return ChatBot(memory, agent_chain)



class ResponseHeaderV1(BaseModel):
    # created_at : str
    text: str