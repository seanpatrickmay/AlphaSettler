#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
.venv/bin/alphasettler fast-loop --dir runs/fast-loop --generations 30 --threads 10 >> runs/fast-loop.log 2>&1
