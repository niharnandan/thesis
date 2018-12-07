#!/bin/sh

module purge
module load python/3.6.5
python coupled_model.py 50000

