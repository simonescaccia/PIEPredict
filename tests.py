import sys
import tensorflow as tf
from pie_data import PIE
import yaml

def check_venv():
    return sys.prefix != sys.base_prefix

def check_gpu():
    return "Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU')))

# print("Check if running in virtual environment: ", check_venv())
# print("Check if GPU is available: ", check_gpu())


with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

imdb = PIE(data_path=config_file['PIE_PATH'])
#imdb.extract_images_and_save_features()
imdb.organize_features()