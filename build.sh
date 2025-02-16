#!/usr/bin/env bash

set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
