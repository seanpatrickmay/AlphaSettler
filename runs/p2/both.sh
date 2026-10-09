#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
bash runs/p2/arm.sh A && bash runs/p2/arm.sh B --gumbel 100
echo "=== both exit $?" >> runs/p2/both.out
