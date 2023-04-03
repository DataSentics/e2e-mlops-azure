#!/bin/bash

conda env create -f environment.yml -p .venv
conda activate $PWD/.venv

pip install -r requirements.txt
