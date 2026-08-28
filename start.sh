#!/bin/bash

uvicorn app.main:app --host 127.0.0.1 --port 8000 &

python app/gradio_app.py
