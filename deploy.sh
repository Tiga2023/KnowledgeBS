#!/bin/bash

# Exit early on errors
set -eu

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx
  
 # Set working directory
WORKDIR /app

# Expose the Streamlit port
EXPOSE 8501

# Python buffers stdout. Without this, you won't see what you "print" in the Activity Logs
export PYTHONUNBUFFERED=true

# Install Python 3 virtual env
VIRTUALENV=.data/venv

if [ ! -d $VIRTUALENV ]; then
  python3 -m venv $VIRTUALENV
fi

if [ ! -f $VIRTUALENV/bin/pip ]; then
  curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | $VIRTUALENV/bin/python
fi

# Install the requirements
$VIRTUALENV/bin/pip install -r requirements.txt

# Run a glorious Python 3 server
$VIRTUALENV/bin/python3 app.py
