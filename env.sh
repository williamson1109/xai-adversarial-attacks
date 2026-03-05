#!/bin/bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv/bin/activate
python --version
python -m pip --version
