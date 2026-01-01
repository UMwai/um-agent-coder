import sys
import threading
import time
import itertools
from typing import Optional
from um_agent_coder.utils.colors import ANSI

class Spinner:
    """Threaded spinner for CLI progress indication."""
    def __init__(self, text: str = "Loading...", delay: float = 0.1, verbose: bool = True):
        self.text = text; self.delay = delay; self.verbose = verbose
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.running = False; self.thread: Optional[threading.Thread] = None
        self._is_tty = sys.stdout.isatty()
        self.start_time = None

    def start(self):
        if not self.verbose: return
        self.start_time = time.time()
        if not self._is_tty:
            print(f"{self.text}..." if not self.text.endswith("...") else self.text)
            return
        self.running = True
        self.thread = threading.Thread(target=self._spin); self.thread.daemon = True
        self.thread.start()

    def stop(self, success: bool = True):
        if not self.verbose: return
        self.running = False
        if self.thread: self.thread.join()

        # Calculate elapsed time
        elapsed_str = ""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed >= 0.1:  # Only show if it took significant time
                elapsed_str = f" ({elapsed:.1f}s)"

        if self._is_tty:
            symbol = ANSI.style('✓', ANSI.GREEN) if success else ANSI.style('✗', ANSI.FAIL)
            time_display = ANSI.style(elapsed_str, ANSI.BLUE) if elapsed_str else ""
            sys.stdout.write(f"\r{symbol} {self.text}{time_display}\033[K\n"); sys.stdout.flush()

    def update(self, text: str):
        self.text = text

    def _spin(self):
        while self.running:
            sys.stdout.write(f"\r{ANSI.style(next(self.spinner), ANSI.CYAN)} {self.text}\033[K")
            sys.stdout.flush(); time.sleep(self.delay)

    def __enter__(self):
        self.start(); return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop(success=exc_type is None)
