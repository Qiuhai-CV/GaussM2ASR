#!/usr/bin/env bash
set -e

pip install -r requirements.txt
BASICSR_EXT=True python setup_basicsr.py develop