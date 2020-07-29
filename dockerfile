FROM tensorflow/tensorflow:2.2.0

MAINTAINER hsh2438@naver.com

COPY . /workspace

WORKDIR /workspace

RUN pip install -r requirements.txt

RUN chmod +x start.sh