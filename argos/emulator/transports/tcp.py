import socket
import time
import logging


class TCPTransport:

    def __init__(
        self,
        host="127.0.0.1",
        port=9000,
        reconnect_delay=3
    ):

        self.logger = logging.getLogger("emulator.transport.tcp")

        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay

        self.sock = None

        self.connect()

    def connect(self):

        while True:

            try:

                self.logger.info(
                    f"Connecting to {self.host}:{self.port}"
                )

                self.sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM
                )

                self.sock.connect(
                    (self.host, self.port)
                )

                self.logger.info("Connected")

                return

            except Exception as e:

                self.logger.warning(
                    f"Connection failed: {e}"
                )

                self.logger.info(
                    f"Retrying in {self.reconnect_delay}s"
                )

                time.sleep(self.reconnect_delay)

    def send(self, msg: str):

        try:

            # newline-delimited JSON
            payload = msg + "\n"

            self.sock.sendall(
                payload.encode()
            )

        except Exception as e:

            self.logger.warning(
                f"Send failed: {e}"
            )

            self.close()

            self.logger.info(
                "Attempting reconnect"
            )

            self.connect()

            # retry once after reconnect
            payload = msg + "\n"

            self.sock.sendall(
                payload.encode()
            )

    def close(self):

        if self.sock:

            try:
                self.sock.close()

            except Exception:
                pass

            self.sock = None

