FROM pytorch/torchserve:0.3.0-cpu


COPY requirements_deploy.txt requirement.txt

RUN pip install -r requirement.txt --no-cache-dir

RUN dvc pull model_store

USER root

CMD "torchserve --start --model-store model_store --models distilbert=distilbert.mar"

     
