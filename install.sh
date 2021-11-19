#!/usr/bin/env bash

# Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
# Install script for Python environment.
# Organisation: Brno University of Technology - Faculty of Information Technology
# Author: Daniel Konecny (xkonec75)
# Date: 19. 11. 2021

PYTHON="python3"
VENV=".env"

test -d $VENV || $PYTHON -m venv $VENV
source "$VENV/bin/activate"
pip install -r requirements.txt
