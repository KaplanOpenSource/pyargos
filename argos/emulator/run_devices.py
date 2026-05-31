#! /usr/bin/env python
import argparse
import json
import logging
from devices.sonic import deviceSonic
from transports.my_kafka import KafkaTransport
from transports.tcp import TCPTransport

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("emulator.run_device")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deviceName",
        required=True,
        help="Device name"
    )

    parser.add_argument(
        "--frequency",
        required=True,
        type=float,
        help="Frequency in Hz"
    )

    parser.add_argument(
        "--duration",
        required=True,
        help="Duration string (example: 10s, 5m, 1h)"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic"
    )

    parser.add_argument(
        "--bootstrapServers",
        default="127.0.0.1:9092",
        help="Kafka bootstrap servers"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP server host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="TCP server port"
    )

    args = parser.parse_args()

    logger.info("Starting emulator")

    device = deviceSonic(
        deviceName=args.deviceName,
        frequency=args.frequency,
        duration=args.duration
    )



    transport = KafkaTransport()

    try:

        for msg in device.stream_messages():

            sndmsg = json.dumps(msg, default=str)

            logger.info(
                f"[{args.deviceName}] Sending message: {sndmsg}"
            )

            transport.send(
                msg=sndmsg
            )

    except KeyboardInterrupt:
        logger.info("Stopping emulator")

    finally:
        transport.close()
        logger.info("Transport closed")

"""
"""