#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install it and try again."
    exit
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found. Please install it and try again."
    exit
fi

# Install the required Python packages
pip3 install -r requirements.txt

# Run the game
python3 q_gui.py