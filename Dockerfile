# FROM tensorflow/tensorflow:2.6.0
FROM python:3.8
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/vision-api-key.json"
WORKDIR /app
RUN ls
COPY ds_api_server/requirements.txt requirements.txt
COPY ds_api_server/install.sh .
# COPY langchain_gradio/install.sh .
# RUN pip install -U Cython cmake numpy
RUN pip install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
RUN apt-get update && apt-get install ffmpeg  -y
# CMD [ "python3","main.py"]
COPY . .
# Make scripts executable
RUN chmod +x /app/install.sh && /app/install.sh
WORKDIR /app/ds_api_server
