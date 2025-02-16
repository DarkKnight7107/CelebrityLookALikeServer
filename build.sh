#!/bin/bash
apt-get update && apt-get install -y cmake g++ libopenblas-dev liblapack-dev libx11-dev
pip install dlib
pip install -r requirements.txt
