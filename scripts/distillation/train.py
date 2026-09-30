"""Run Slime unchanged, from its own working directory."""
import os
from pathlib import Path
import runpy
import sys

root = Path(os.environ['SLIME_ROOT']).expanduser().resolve()
os.chdir(root)
sys.path.insert(0, str(root))
sys.argv[0] = str(root / 'train.py')
runpy.run_path(sys.argv[0], run_name='__main__')
