import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
def use(folder):
    p = str(ROOT / folder)
    if p not in sys.path:
        sys.path.insert(0, p)
