#!/bin/bash
git pull
RUN pip install -r requirements.txt --no-cache-dir
dvc pull