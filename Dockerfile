FROM python:3.9.13-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.7.1 
RUN pip install torchvision>=0.8.1
WORKDIR /app
COPY . .