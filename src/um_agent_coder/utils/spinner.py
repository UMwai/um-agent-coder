import sys
import threading
import time
import itertools
from typing import Optional

class Spinner:
    """
    A simple thread-based spinner for CLI loading states.

    Usage:
        with Spinner("Processing..."):
            long_running_task()
    """

    def __init__(self, message: str = "Loading...", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def spin(self):
        """The spinner animation loop."""
        while self.running:
            with self.lock:
                sys.stdout.write(f"\r{next(self.frames)} {self.message}")
                sys.stdout.flush()
            time.sleep(self.delay)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join()

        # Clear the spinner line
        with self.lock:
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
            sys.stdout.flush()

    def update(self, message: str):
        """Update the spinner message while running."""
        with self.lock:
            # clear previous message length if new one is shorter to avoid artifacts
            if len(message) < len(self.message):
                 sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
            self.message = message
