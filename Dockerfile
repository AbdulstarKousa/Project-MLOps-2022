FROM ubuntu:latest

ENV TZ=Europe/Copenhagen \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip\
    git

WORKDIR /
RUN pip install --upgrade pip
RUN git clone https://github.com/AbdulstarKousa/Project-MLOps-2022.git
WORKDIR Project-MLOps-2022/
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT dvc pull && /bin/bash