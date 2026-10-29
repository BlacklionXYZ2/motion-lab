"""Tests in this folder import this version's own motionlab package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
