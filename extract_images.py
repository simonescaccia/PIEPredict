from pie_data import PIE
import os
import yaml

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

pie_path = config_file['PIE_PATH']

os.chdir(pie_path)

imdb = PIE(data_path=pie_path)
imdb.extract_and_save_images(extract_frame_type='annotated')
imdb.organize_features()