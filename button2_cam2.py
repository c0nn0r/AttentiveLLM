#!/usr/bin/env python3
"""
https://github.com/c0nn0r/AttentiveLLM
Connor Dickie August 2025


Mouse Click Controller for Raspberry Pi
---------------------------------------

Listens on TCP port 4212 for the strings **play** or **pause**.
Whenever either command is received, it “clicks” a mouse button that
you have hard-wired to a Pi GPIO pin.

Hardware wiring
===============

* INSIDE THE MOUSE – identify the two pads of the switch you want to spoof.
  ▸ One pad is tied to battery negative → **GND pad**  
  ▸ Other pad floats high via the mouse’s pull-up → **SIGNAL pad**

* RASPBERRY PI
  ▸ GND pad  → any Pi **GND** header pin (e.g. physical pin 20)  
  ▸ SIGNAL pad → chosen **GPIO** (default BCM 18 = physical pin 12)

Operation
=========

* Button released → GPIO is INPUT (high-impedance)  
* Button pressed  → GPIO is OUTPUT and driven **LOW**

We *never* drive the pin HIGH, so the Pi does **not** push 3 V3 into the
mouse; it only sinks a few hundred µA to ground.

-----------------------------------------------------------
"""

import logging
import socket
import sys
import time
import signal

try:
    import RPi.GPIO as GPIO
except ImportError:                           # allow dev on non-Pi host
    GPIO = None
    print("RPi.GPIO not found – running in simulation mode")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
TCP_PORT        = 5001          # port to listen on
GPIO_PIN        = 13            # BCM numbering (physical header pin 12)
CLICK_DURATION  = 0.10          # seconds the button stays “down”

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MouseClickCtl")

# ---------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------
class MouseClickController:
    def __init__(self,
                 gpio_pin : int  = GPIO_PIN,
                 tcp_port : int  = TCP_PORT,
                 hold_sec : float= CLICK_DURATION):

        self.gpio_pin   = gpio_pin
        self.tcp_port   = tcp_port
        self.hold_sec   = hold_sec
        self.running    = False
        self.sock       = None

        # --- GPIO setup ------------------------------------------------
        if GPIO is None:
            log.warning("GPIO unavailable – all clicks will be simulated")
            return

        GPIO.setmode(GPIO.BCM)
        # start up in released state (input / high-Z)
        GPIO.setup(self.gpio_pin, GPIO.IN)
        log.info("GPIO %d configured (INPUT → released state)", self.gpio_pin)

    # ----------------- GPIO helpers -----------------------------------
    def _press(self):
        if GPIO is not None:
            GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
        log.debug("Mouse button DOWN")

    def _release(self):
        if GPIO is not None:
            GPIO.setup(self.gpio_pin, GPIO.IN)
        log.debug("Mouse button UP")

    def click(self):
        """One full down-up cycle."""
        self._press()
        time.sleep(self.hold_sec)
        self._release()
        log.info("Click complete (held %.0f ms)", self.hold_sec*1000)

    # ----------------- TCP server loop --------------------------------
    def start(self):
        self.running = True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("", self.tcp_port))
            self.sock.listen(1)
            log.info("Listening on TCP port %d …", self.tcp_port)

            while self.running:
                log.info("Waiting for connection …")
                conn, addr = self.sock.accept()
                log.info("Client %s connected", addr)

                with conn:
                    data = b""
                    while self.running:
                        chunk = conn.recv(1024)
                        if not chunk:
                            break
                        data += chunk
                        # handle line-oriented commands
                        while b"\n" in data:
                            line, data = data.split(b"\n", 1)
                            self._handle_cmd(line.decode().strip(), conn)

                log.info("Client disconnected")

        except Exception as e:
            if self.running:
                log.error("Server error: %s", e)

        finally:
            self.stop()

    def _handle_cmd(self, cmd: str, conn):
        log.info("Received: %s", cmd)
        if cmd.lower() in ("play", "pause"):
            self.click()
            conn.sendall(b"OK\n")
        else:
            conn.sendall(b"ERR unknown command\n")

    # ----------------- cleanup ----------------------------------------
    def stop(self):
        if not self.running:
            return
        log.info("Shutting down…")
        self.running = False

        if self.sock:
            self.sock.close()

        if GPIO is not None:
            GPIO.cleanup()
            log.info("GPIO cleaned up")

# ---------------------------------------------------------------------
# Graceful shutdown handlers
# ---------------------------------------------------------------------
def _shutdown(signum, frame, ctl: MouseClickController):
    log.info("Signal %d received", signum)
    ctl.stop()
    sys.exit(0)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    ctl = MouseClickController()

    # trap Ctrl-C / SIGTERM
    signal.signal(signal.SIGINT,  lambda s,f: _shutdown(s, f, ctl))
    signal.signal(signal.SIGTERM, lambda s,f: _shutdown(s, f, ctl))

    try:
        ctl.start()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()

if __name__ == "__main__":
    main()

