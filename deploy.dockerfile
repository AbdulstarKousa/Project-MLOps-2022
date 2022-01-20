FROM pytorch/torchserve:0.3.0-cpu

COPY model_store/ model_store/
COPY requirements_deploy.txt requirement.txt

RUN pip install -r requirement.txt --no-cache-dir

USER root

CMD "torchserve --start --model-store model_store --models distilbert=distilbert.mar"

     
