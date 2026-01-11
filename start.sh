#!/bin/bash

echo "Installing dependencies..."
./.venv/bin/pip3 install -r requirements.txt

echo "Starting Kafka..."
docker-compose up -d
sleep 10

echo "Starting Spark Streaming..."
./.venv/bin/python3 spark/online-stream.py > spark.log 2>&1 &
sleep 5

echo "Starting Consumer..."
./.venv/bin/python3 consume_messages.py > consumer.log 2>&1 &
sleep 3

echo "Starting Producer..."
./.venv/bin/python3 produce_messages.py