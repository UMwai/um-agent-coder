import sys
import threading
import time
from typing import Optional
from um_agent_coder.utils.colors import ANSI

class Spinner:
    """A thread-based spinner for CLI feedback."""

    # Spinner frames
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Loading...", delay: float = 0.1):
        self.message = message
        self.delay = delay
        self.stop_running = False
        self.spin_thread: Optional[threading.Thread] = None
        self.is_tty = sys.stdout.isatty()

    def spin(self):
        """Spin the spinner until stopped."""
        i = 0
        while not self.stop_running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            # Clear line and print spinner
            sys.stdout.write(f"\r{ANSI.style(frame, ANSI.CYAN)} {self.message}")
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1

    def start(self):
        """Start the spinner thread."""
        if self.is_tty:
            self.stop_running = False
            self.spin_thread = threading.Thread(target=self.spin)
            self.spin_thread.daemon = True # Ensure it dies if main thread dies
            self.spin_thread.start()
        else:
            # If not TTY, just print the message once
            print(f"{self.message}...")

    def stop(self, success: bool = True, final_message: Optional[str] = None):
        """Stop the spinner and print final message."""
        if self.is_tty:
            self.stop_running = True
            if self.spin_thread:
                self.spin_thread.join()

            # Clear the line
            sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
            sys.stdout.flush()

            if final_message:
                color = ANSI.GREEN if success else ANSI.FAIL
                symbol = "✓" if success else "✗"
                print(f"{ANSI.style(symbol, color)} {final_message}")
            elif final_message is None:
                # If no final message, assume we just want to clear or leave it
                # If we want to leave the message as "done", we can print it.
                color = ANSI.GREEN if success else ANSI.FAIL
                symbol = "✓" if success else "✗"
                print(f"{ANSI.style(symbol, color)} {self.message}")
        else:
            if final_message:
                print(final_message)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.stop(success=False)
        else:
            self.stop(success=True)
