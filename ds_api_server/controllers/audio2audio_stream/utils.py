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
import json
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
        t1 = time.time()
        # print("token aaya")
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
        self.user_detail=mongo_connect(os.environ.get('MONGO_CONN',''),'user_details')

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
    


    def conversation(self,text,selected_model,user_id,context_enable,queue,session_id,start_date="2022-01-01",end_date="2024-01-01"):
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
            logger.info("conversation object initiated");t1=time.time()

            chat = ChatOpenAI(temperature=0,model_name=selected_model,streaming=True,callback_manager=CallbackManager([MyCustomHandler(queue)]));print(time.time()-t1,"*"*10,"open ai connection established")
            valid_session = self.mongocon_ds.db.conversations_collection.find_one({"session_id":session_id})
            session_id = session_id if valid_session is not None else "None"
            print(session_id)
            base_temp = BASE_TEAMPLATE.format(session_id)
            print(base_temp)
            print(time.time()-t1,"*"*10,"base template")
            result = self.user_detail.db.bank_info.find_one({"user_id":user_id})
            if result:
                logger.info("bank info found")
                account_number = result["account_no"]
                current_balance = result["current_balance"]
                balance_string = "account Number : {}  Available Balance : {}".format(account_number,current_balance)
                template = base_temp+"\nPrevious Conversation Context : \n"+context+"\nUser bank details information below : \n"+balance_string+"\n"+"""
                Question: {text}
                Answer:
                """
            else:
                logger.info("bank info not found")
                template = base_temp+context+"""
                Question: {text}
                Answer:
                """
            # redis_connect.get(user_id)
            # redis_connect.set('some_key', context)
            logger.info("passing context and user input to llm")
            prompt_template = PromptTemplate(input_variables=["text"], template=template,validate_template=False)
            answer_chain = LLMChain(llm=chat, prompt=prompt_template)
            logger.info("returning llm output")
            return answer_chain
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
            raise
            return {"msg":str(e)}

    def update_context(self,text,answer,user_id,session_id):
        try:
            logger.info("updating context")
            context_dict = {"user_id":user_id,
                            "human":text,
                            "bot":answer,
                            "session_id":session_id,
                            "created_at":datetime.now()}
            self.mongocon_ds.db.conversations_collection.insert_one(context_dict)
            logger.info("context updated")
            return True 
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
        



class RequestHeaderV1(BaseModel):
    # created_at : str
    text: Optional[str] = None


class RequestHeaderV2(BaseModel):
    # created_at : str
    text: str
    voice_code: str = 'en-IN'
    voice_gender : str = 'FEMALE'
    voice_name: str = 'en-IN-Standard-A'