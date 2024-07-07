#!/bin/bash
export FLASK_APP=client.client
export SERVER_URL=http://localhost:5000
python3 -m flask run --host=0.0.0.0 --port=5001

