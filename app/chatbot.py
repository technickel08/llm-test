import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)

import gradio as gr
import openai
import os
from functools import partial
import time
import os
# import openai
import whisper
from gtts import gTTS
os.environ["OPENAI_API_KEY"] = "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"
import chat_agent
from langchain.schema import (
    HumanMessage
)

model = None
# whisper.load_model("medium")

def set_api_key(api_key):
    openai.api_key = api_key
    logger.info("openai api key - {}".format(api_key))
    # os.environ["OPENAI_API_KEY"] = "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"
    # "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"
    return "API Key set successfully."

def get_response(chatbot, 
                 api_key,
                 selected_model,
                 audio,
                 conversation_history="",
                 ):
    logger.info("get response initialised")
    set_api_key(api_key)

    # Preserve the memory of the current chatbot
    preserved_memory = chatbot.memory
    logger.info("chatbot memory initialised")
    if audio is not None:
        logger.info("Audio file found, API CALL function execution start")
        logger.info("Audio File path - {}".format(audio))
        user_input,_,audio_file = api_call(audio)
    else:
        logger.info("Audio file not found")
        user_input="Hey"
    print(user_input)
    # Create a new chat agent based on the selected model and seed the memory
    chatbot = chat_agent.create_chatbot(model_name=selected_model, seed_memory=preserved_memory)

    # Get raw chat response
    response = chatbot.agent.run(user_input).strip()

    # Iterate through messages in ChatMessageHistory and format the output
    updated_conversation = '<div style="background-color: hsl(30, 100%, 30%); color: white; padding: 5px; margin-bottom: 10px; text-align: center; font-size: 1.5em;">Chat History</div>'
    for i, message in enumerate(chatbot.memory.chat_memory.messages):
        if isinstance(message, HumanMessage):
            prefix = "User: "
            background_color = "hsl(0, 0%, 40%)"  # Dark grey background
            text_color = "hsl(0, 0%, 100%)"  # White text
        else:
            prefix = "Chatbot: "
            background_color = "hsl(0, 0%, 95%)"  # White background
            text_color = "hsl(0, 0%, 0%)"  # Black text
        updated_conversation += f'<div style="color: {text_color}; background-color: {background_color}; margin: 5px; padding: 5px;">{prefix}{message.content}</div>'
    myobj = gTTS(text=message.content, lang='en', slow=False) 
    myobj.save("response.wav")
    return "response.wav"

def speech_to_text(audio):
    return "found" if audio is not None else "not found"


def api_call(file):
    logger.info("Whisper API call initialised")
    audio_file= open(file, "rb")
    logger.info("audo file loaded")
    t1 = time.time()
    result = openai.Audio.transcribe("whisper-1",audio_file)
    logger.info("Whisper API inference compelete - {}s".format(round(time.time()-t1)))
    out = result.text
    myobj = gTTS(text=out, lang='en', slow=False) 
    myobj.save("test.wav")
    logger.info("API call Function executed")
    return result.text,out,"test.wav"

def transcribe(audio):
    # load audio and pad/trim it to fit 30 seconds
    start=time.time()
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    w1=time.time()
    print("loaded audion in {} sec".format(time.time()-start))

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    w2=time.time()
    
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    #options = whisper.DecodingOptions(fp16 = False)
    options = whisper.DecodingOptions(fp16 = False)
    
    result = whisper.decode(model, mel, options)
    print("audio-{}".format(result.text))
    w3=time.time()
    print(w3-w2)
    #out=db_chain.run(result.text)
    out=result.text
    w4=time.time()
    print(w4-w3)
    myobj = gTTS(text=out, lang='en', slow=False) 
    myobj.save("test.wav") 
    w5=time.time()
    print(w5-w4)
    # result = openai.Audio.transcribe("whisper-1",audio)
    # out = result.text
    # myobj = gTTS(text=out, lang='en', slow=False) 
    # myobj.save("test.wav")
    return result.text,out,"test.wav"

def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    api_key_input = gr.components.Textbox(
        lines=1,
        label="Enter OpenAI API Key",
        value=api_key,
        type="password",
    )

    model_selection = gr.components.Dropdown(
        choices=["gpt-4", "gpt-3.5-turbo"],
        label="Select a GPT Model",
        value="gpt-3.5-turbo",
    )

    audio_input = gr.components.Audio(
        source="microphone",
        type="filepath",
        # streaming="microphone",
        label="Please speaker to record",
    )

    user_input = gr.components.Textbox(
        lines=3,
        label="Enter your message",
    )

    output_history = gr.outputs.HTML(
        label="Updated Conversation",
    )

    output_history_2 = gr.components.Textbox(
        lines=1,
        label="Enter OpenAI API Key",
        value=api_key,
        type="password",
    )

    chatbot = chat_agent.create_chatbot(model_name=model_selection.value)

    inputs = [
        # api_key_input,
        # model_selection,
        audio_input,
        user_input,
    ]

    # iface = gr.Interface(
    #     fn=partial(get_response, chatbot),
    #     inputs=inputs,
    #     outputs=[output_history],
    #     title="LiveQuery GPT-4",
    #     description="A simple chatbot using GPT-4 and Gradio with conversation history",
    #     allow_flagging="never",
    # )

    # iface.launch(server_name="0.0.0.0",share=False)
    iface = gr.Interface(
    title = 'DEMO', 
    fn=partial(get_response, chatbot), 
    inputs=[
        api_key_input,
        model_selection,
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "audio"
    ],
    live=False).launch(server_name="0.0.0.0",share=False)


if __name__ == "__main__":
    main()