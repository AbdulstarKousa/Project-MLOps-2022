FROM python:3.9-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY prep.py prep.py
COPY test_environment.py test_environment.py
COPY Makefile Makefile
COPY src/ src/
COPY data/ data/
COPY .dvc/ .dvc/


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
#RUN dvc pull

ENTRYPOINT ["python", "-u", "prep.py"]