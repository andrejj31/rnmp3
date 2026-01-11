import json
import pandas as pd
import time
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
LABEL_COL = "Diabetes_012"
DATA_PATH = "/Users/andrej/Desktop/rnmp/RNMP_homework1/data/online.csv"
TOPIC_NAME = "health-data"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    security_protocol="PLAINTEXT"
)

df = pd.read_csv(DATA_PATH)

for idx, row in df.iterrows():
    data = row.drop(LABEL_COL).to_dict()
    producer.send(TOPIC_NAME, value=data)
    print(f"✅ Sent row {idx}: {data}")
    time.sleep(0.1)

producer.flush()
producer.close()
print(f"🎉 All data sent to Kafka topic: {TOPIC_NAME}")
