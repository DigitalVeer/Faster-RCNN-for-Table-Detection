"""Make the repository root importable so ``import frcnn`` works under pytest
without needing an editable install first."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
