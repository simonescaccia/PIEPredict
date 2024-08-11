from genericpath import isdir, isfile
from ntpath import join
import os
import shutil
import sys
import cv2
import tensorflow as tf
from pie_data import PIE
import yaml
from keras.utils import load_img
from PIL import Image, ImageChops

def check_venv():
    return sys.prefix != sys.base_prefix

def check_gpu():
    return "Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU')))

def check_images(config_file):
    pie_clips_path = config_file['PIE_RAW_PATH']
    pie_clips_path = os.path.join(pie_clips_path, 'set01')
    vid = 'video_0001.mp4'
    temp_folder = 'test_image'
    temp_folder_path = os.path.join(os.getcwd(), temp_folder)
    frame_num = 1
    
    # Extract the images from the video clips
    vidcap = cv2.VideoCapture(join(pie_clips_path, vid))
    success, image = vidcap.read()
    if not success:
        print('Failed to open the video {}'.format(vid))
        return
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    image_path = join(temp_folder_path, "%05.f.png") % frame_num
    print("Saving image to: ", image_path)
    if not isfile(image_path):
        print(cv2.imwrite(image_path, image))
    
    image1 = load_img(image_path)
    # CV image to PIL image
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image2 = Image.fromarray(image2)
    
    # Compare the two images
    diff = ImageChops.difference(image1, image2)
    if diff.getbbox():
        print("images are different")
    else:
        print("images are the same")

    # Delete the temporary folder
    shutil.rmtree(temp_folder_path)


with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

# print("Check if running in virtual environment: ", check_venv())
# print("Check if GPU is available: ", check_gpu())

# imdb = PIE(data_path=config_file['PIE_PATH'])
# imdb.extract_images_and_save_features()
# imdb.organize_features()

check_images(config_file)