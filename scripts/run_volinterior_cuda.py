#!/usr/bin/env python3
"""Source-checkout entry point; same options as the installed volinterior-cuda command."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volinterior_cuda.cli import main

if __name__ == "__main__":
    main()
