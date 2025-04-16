from pie_data import PIE
import os
import yaml

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

pie_path = config_file['PIE_PATH']
pie_raw_path = config_file['PIE_RAW_PATH']

os.chdir(pie_path)

imdb = PIE(data_path=pie_path, pie_raw_path=pie_raw_path)
imdb.extract_and_save_images()