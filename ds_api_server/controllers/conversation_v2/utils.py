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
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient,ReplaceOne
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager,CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from connections.mongo_connect import mongo_connect
from langchain.prompts import PromptTemplate
from datetime import datetime
from baseTemplates.baseTemplate import BASE_TEAMPLATE
from langchain.schema import LLMResult

    

# class MyCustomHandler(BaseCallbackHandler):
#     # def __init__(self,queue):
#     #     super(MyCustomHandler,self).__init__()
#     #     print("custom handler initiated")
#     #     self.queue = queue
#     def on_llm_new_token(self,token,**kwargs):
#         print(token)
#         self.queue.put(token)
#         yield token


class MyCustomHandler(BaseCallbackHandler):
    def __init__(self,queue):
        super(MyCustomHandler,self).__init__()
        print("custom handler initiated")
        self.queue = queue
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # print(f"My custom handler, token: {token}")
        self.queue.put(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        self.queue.put("DONE")


class ChatBot:
    def __init__(self):
        self.load_db()
        # self.memory = memory
        # self.agent = agent_chain


    def load_db(self):
        self.mongocon_ds=mongo_connect(os.environ.get('MONGO_CONN',''),'ds_db')

    def load_context(self,user_id,start_date,end_date):
        try:
            logger.info("retriving context from db")
            context = ""
            # cont = self.mongocon_ds.get_context_vars(user_id)
            # self.mongocon_ds.db.conversations_collection()
            start_timestamp = datetime.strptime(start_date, '%Y-%m-%d')
            end_timestamp = datetime.strptime(end_date, '%Y-%m-%d')
            result = self.mongocon_ds.db.conversations_collection.find({"user_id":user_id})
            if result is not None:
                # topValue = sorted(list(result), key=lambda k: k['created_at'],reverse=True)
                topValue = list(result)
                temp_string = []
                for x in topValue:
                    human = x["human"]
                    bot = x["bot"]
                    created_at = x.get('created_at',None)
                    # created_at = x["created_at"]
                    # print(created_at,type(created_at))
                    # print(x.keys())
                    logger.info("line 65")
                    temp_string.append("datetime : {} \n human : {} \n bot : {} \n".format(str(created_at),human,bot))
                    logger.info("line 67")
                print(temp_string)
                context = context.join(temp_string)
            logger.info("context retrival complete from db")
            # print(context)
            return context,topValue
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
            return None
    

    def conversation(self,text,selected_model,user_id,context_enable,queue,start_date="2022-01-01",end_date="2024-01-01"):
        self.queue = queue
        try:
            if context_enable == True:
                context,context2 = self.load_context(user_id,start_date,end_date)
                if context is not None:
                    context = context[-1000:]
                    print(len(context),"*"*100)
                context = str(context)+"\n"
            else:
                context = "\n"
            logger.info("conversation object initiated")
            text = text.text
            chat = ChatOpenAI(temperature=0,model_name=selected_model,streaming=True,callback_manager=CallbackManager([MyCustomHandler(queue)]))
            base_temp = BASE_TEAMPLATE
            template = base_temp+context+"""
            Question: {text}
            Answer:
            """
            # redis_connect.get(user_id)
            # redis_connect.set('some_key', context)
            logger.info("passing context and user input to llm")
            prompt_template = PromptTemplate(input_variables=["text"], template=template,validate_template=False)
            answer_chain = LLMChain(llm=chat, prompt=prompt_template)
            # answer = answer_chain.run(text)
            # self.update_context(text,answer,user_id)
            logger.info("returning llm output")
            return answer_chain
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
            raise
            return {"msg":str(e)}

    def update_context(self,text,answer,user_id):
        try:
            context_dict = {"user_id":user_id,
                            "human":text,
                            "bot":answer,
                            "created_at":datetime.now()}
            self.mongocon_ds.db.conversations_collection.insert_one(context_dict)
            return True 
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
        



class ResponseHeaderV1(BaseModel):
    # created_at : str
    text: str


class ResponseHeaderV2(BaseModel):
    # created_at : str
    text: str
    language_code: str
    voice_gender : str