import sys

def in_venv():
    return sys.prefix != sys.base_prefix

print(in_venv())