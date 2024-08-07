import sys
import tensorflow as tf

def check_venv():
    return sys.prefix != sys.base_prefix

def check_gpu():
    return "Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU')))

print("Check if running in virtual environment: ", check_venv())
print("Check if GPU is available: ", check_gpu())