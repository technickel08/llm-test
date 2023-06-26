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
from pathlib import Path
from pydub import AudioSegment
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient,ReplaceOne
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager,CallbackManager
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from connections.mongo_connect import mongo_connect
from langchain.prompts import PromptTemplate
from datetime import datetime
from baseTemplates.baseTemplate import BASE_TEAMPLATE
from langchain.schema import LLMResult
from langchain.chains import SimpleSequentialChain
from langchain.agents import load_tools
from langchain.prompts import StringPromptTemplate
from typing import List, Dict, Any
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from typing import List, Union
import re
from langchain.prompts import BaseChatPromptTemplate
from baseTemplates.template import CustomPromptTemplate, read_template
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

# class MyCustomHandler(BaseCallbackHandler):
#     # def __init__(self,queue):
#     #     super(MyCustomHandler,self).__init__()
#     #     print("custom handler initiated")
#     #     self.queue = queue
#     def on_llm_new_token(self,token,**kwargs):
#         print(token)
#         self.queue.put(token)
#         yield token


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class MyCustomHandler(FinalStreamingStdOutCallbackHandler):
    def __init__(self,queue):
        super(MyCustomHandler,self).__init__()
        print("custom handler initiated","*"*20)
        self.queue = queue
        self.answer = False
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        t1 = time.time()
        # print(token)
        # print(type(self.queue),"*"*10)
        # if token in ["Final","Answer",":"]:
        #     self.answer=True
        # if self.answer == True:
        self.queue.put(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        self.queue.put("DONE")


# Import necessary classes and modules
# Define the CustomOutputParser class, inheriting from AgentOutputParser
class CustomOutputParser(AgentOutputParser):
    
    # Override the parse method from the parent class
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print("parsing output")
        # If the "Final Answer:" is found in the output, return an AgentFinish object
        if "Final Answer:" in llm_output:
            print("final answer : ","*"*10)
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Add a condition to handle the case when no action is needed
        if "Action: None" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Thought:")[-1].strip().replace("Action: None", "")},
                log=llm_output,
            )
        # Define the regex pattern to match action and action input from the output
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        # Search for the regex pattern in the llm_output
        match = re.search(regex, llm_output, re.DOTALL)
        # If no match is found, return an AgentFinish object with the original output
        if not match:
            print("no final answer : ","*"*10)
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        # Extract the action and action input from the regex match
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return an AgentAction object with the parsed action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class ChatBot:
    def __init__(self):
        self.load_db()
        # self.memory = memory
        # self.agent = agent_chain
        self.search = DuckDuckGoSearchRun()
        logger.info("Chat search initiated")


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

            # chat = ChatOpenAI(temperature=0,model_name=selected_model,streaming=True,callback_manager=CallbackManager([MyCustomHandler(queue)]))
            print(time.time()-t1,"*"*10,"open ai connection established")
            base_temp = BASE_TEAMPLATE
            print(time.time()-t1,"*"*10,"base template")
            result = self.user_detail.db.bank_info.find_one({"user_id":user_id})
            if result:
                logger.info("bank info found")
                personal_info = json.dumps(result["user_data"]).replace("{","(").replace("}",")")
                # balance_string = "account Number : {}  Available Balance : {}".format(account_number,current_balance)
                template = base_temp+"\nPrevious Conversation Context : \n"+context+"\nUser personal information below in JSON FORMAT : \n"+personal_info+"\n"+"""
                {tools}
                Question: {text}
                Answer:
                {agent_scratchpad}
                """
            else:
                personal_info=""
                logger.info("bank info not found")
                template = base_temp+context+"""
                Question: {text}
                Answer:
                """
            # print(template)
            # redis_connect.get(user_id)
            # redis_connect.set('some_key', context)
            logger.info("passing context and user input to llm")
            output_parser = CustomOutputParser()
            tools = [
        Tool(
            name="Search",
            func=self.search.run,
            description="useful for when you need to answer questions about current events"
        )
        ]
            prompt = CustomPromptTemplate(
                template=read_template(str("/app/ds_api_server/baseTemplates/base.txt")).replace("{session_id}", str(session_id)).\
                                                                                        replace("{chatbot_name}", "Saathi").\
                                                                                        replace("{personal_info}",personal_info),
                tools=tools,
                input_variables=["input", "intermediate_steps"]
            )
            
    # Instantiate a ChatOpenAI object for language model interaction
            llm = ChatOpenAI(temperature=0.8,model_name=selected_model,streaming=True,callback_manager=CallbackManager([MyCustomHandler(queue=queue)]))
            
            # Set up the LLMChain using the ChatOpenAI object and prompt template
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Extract tool names from the tools list
            tool_names = [tool.name for tool in tools]
            # Set up the LLMSingleActionAgent with LLMChain, output parser, and allowed tools
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                return_only_outputs=True
            )

            # Create an AgentExecutor from the agent and tools with verbose output
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
            # agent = initialize_agent(tool, answer_chain, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            # agent = initialize_agent([tool],
            #              chat,
            #              agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #              verbose=True)
            # agent_chain = SimpleSequentialChain(
            #       chains=[answer_chain, agent],
            #       input_key=["text"])
            logger.info("returning llm output")
            return agent_executor
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
            raise
            return {"msg":str(e)}

    def update_context(self,text,answer,user_id):
        try:
            logger.info("updating context")
            context_dict = {"user_id":user_id,
                            "human":text,
                            "bot":answer,
                            "created_at":datetime.now()}
            self.mongocon_ds.db.conversations_collection.insert_one(context_dict)
            logger.info("context updated")
            return True 
        except Exception as e:
            logger.error("some exception occured - {}".format(str(e)))
        



class ResponseHeaderV1(BaseModel):
    # created_at : str
    text: str


class ResponseHeaderV2(BaseModel):
    # created_at : str
    text: str 
    voice_code: str = 'en-IN'
    voice_gender : str = 'FEMALE'
    voice_name: str = 'en-IN-Standard-A'