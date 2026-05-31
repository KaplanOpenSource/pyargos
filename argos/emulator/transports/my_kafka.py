import json
import time
import logging
from kafka import KafkaProducer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ..devices.sonic import deviceSonic


class KafkaTransport:
    def __init__(self, bootstrap_servers="127.0.0.1:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        self.logger = logging.getLogger("emulator.transport.kafka")

    def send(self, topic, msg: dict):
        self.logger.info(f"Sending message: {msg}")
        self.producer.send(topic, msg)

    def close(self):
        self.producer.close()


def run_emulator(args):
    logger = logging.getLogger("emulator.runner")

    logger.info("Creating device")
    device = deviceSonic(
        deviceName=args.deviceName,
        frequency=args.frequency,
        duration=args.duration
    )

    transport = KafkaTransport("127.0.0.1:9092")

    logger.info("Sending messages")

    for msg in device.messages:
        logger.info(f"msg {msg.get('datetime')}: {msg}")
        transport.send(args.topic, msg)
        time.sleep(device.delay_s)

