#!/usr/bin/env python
import argparse
import json
import logging

from devices.sonic import deviceSonic

import csv

class CsvTransport:

    def __init__(self, filename="sonic_data.csv"):
        self.file = open(filename, "a", newline="")
        self.writer = None

    def send(self, topic: str, msg: str):
        data = json.loads(msg)

        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=data.keys()
            )
            self.writer.writeheader()

        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()

"""
class ConsoleTransport:
    
    Simple transport that prints messages instead of sending them anywhere.
    

    def send(self, topic: str, msg: str):
        print(f"[TOPIC: {topic}] {msg}")

    def close(self):
        print("ConsoleTransport closed")
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deviceName",
        default="test_device",
        help="Device name (default: test_device)"
    )

    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        help="Frequency in Hz (default: 1.0)"
    )

    parser.add_argument(
        "--duration",
        default="10s",
        help="Duration string (default: 10s)"
    )

    parser.add_argument(
        "--topic",
        default="test-topic",
        help="Logical topic name (default: test-topic)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("emulator.run_device")

    args = parse_args()

    logger.info("Starting emulator (LOCAL MODE - no Kafka)")

    device = deviceSonic(
        deviceName=args.deviceName,
        frequency=args.frequency,
        duration=args.duration
    )

    transport = CsvTransport("sonic_data.csv")
    try:
        for msg in device.stream_messages():

            sndmsg = json.dumps(msg, default=str)

            logger.info(
                f"[{args.deviceName}] Sending message: {sndmsg}"
            )

            transport.send(
                topic=args.topic,
                msg=sndmsg
            )

    except KeyboardInterrupt:
        logger.info("Stopping emulator")

    finally:
        transport.close()
        logger.info("Transport closed")


