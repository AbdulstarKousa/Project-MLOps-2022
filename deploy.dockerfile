FROM pytorch/torchserve:0.3.0-cpu

USER root

COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY .git/ .git/
COPY requirements_deploy.txt requirement.txt
COPY model_store.dvc model_store.dvc

RUN pip install --upgrade pip
RUN pip install -r requirement.txt --no-cache-dir

RUN dvc pull model_store



CMD "torchserve --start --model-store model_store --models distilbert=distilbert.mar"

     
