#!/bin/bash

# activate virtualenv
source /neuroenv/bin/activate

# add module to python path
export PYTHONPATH=/code:$PYTHONPATH

# chaing to main directory
cd /code

if [[ -z "$@" ]]; then
    ipython
else
    python "$@"
fi
