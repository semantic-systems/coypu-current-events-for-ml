from pathlib import Path
from os.path import abspath, split

ml_module_dir = Path(split(abspath(__file__))[0])
