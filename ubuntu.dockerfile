FROM ubuntu:latest
# Update package manager (apt-get) 
# and install (with the yes flag `-y`)
# Python and Pip

ENV TZ=Europe/Copenhagen \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip\
    git

#COPY requirements.txt requirements.txt
#COPY setup.py setup.py
#COPY prep.py prep.py
#COPY test_environment.py test_environment.py
#COPY Makefile Makefile
#COPY src/ src/
#COPY .dvc/ .dvc/


WORKDIR /
RUN pip install --upgrade pip
RUN git clone https://github.com/AbdulstarKousa/Project-MLOps-2022.git
WORKDIR Project-MLOps-2022/
RUN pip install -r requirements.txt --no-cache-dir
RUN dvc pull

CMD ["bash"]
#ENTRYPOINT ["python3"]