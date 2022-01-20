Team-X
==============================

A short description of the project:
- This is the group repo for [02476 Machine Learning Operation](https://kurser.dtu.dk/course/02476) Jan. 2022. 
- Course [Home page](https://skaftenicki.github.io/dtu_mlops/)
- FrameWork [Transformers](https://github.com/huggingface/transformers) 
- To get setup with both code and data by simply running:
    - `git clone https://github.com/AbdulstarKousa/Project-MLOps-2022.git`
    - `dvc pull`
- To pull the docker image from the gcp server:
    - `docker pull gcr.io/gcr.io/team-x-338109/main:latest`
- To run the docker image interactively:
    - `docker run -it gcr.io/gcr.io/team-x-338109/main:latest`

In this project we make use of the [Hugging Face Transformer](https://huggingface.co/docs/transformers) framework to create a binary sentiment classifier for review’s given to highly polarized movies on [IMDb](https://www.imdb.com).
The dataset is available on the [Hugging Face Github](https://github.com/huggingface/datasets/tree/master/datasets/imdb) page.

The model used for performing the classification is a **Bidirectional Encoder Representations from Transformers** also known as [**BERT**](https://arxiv.org/abs/1810.04805).
Transfer learning will be applied by training a pre-trained model on 25.000 positive/negative labeled movie reviews, and then evaluated on another 25.000 reviews.
Different configurations of BERT will be trained and evaluated.

The whole pipeline is also available as a Docker-Image:
  
---
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
