from genericpath import isdir, isfile
from ntpath import join
import os
import shutil
import sys
import cv2
import numpy as np
from prettytable import PrettyTable
import tensorflow as tf
from pie_data import PIE
import yaml
from keras.utils import load_img
from PIL import Image, ImageChops
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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
    
    # Compare the two images #TODO compare the input to the vgg16 model
    diff = ImageChops.difference(image1, image2)
    if diff.getbbox():
        print("Images are different")
    else:
        print("Images are the same")

    # Delete the temporary folder
    shutil.rmtree(temp_folder_path)

def print_intent_results(path):
    path = os.path.join(path, 'ped_intents.pkl')
    obj = pd.read_pickle(path)

    acc = accuracy_score(obj['gt'], np.round(obj['results']))
    f1 = f1_score(obj['gt'], np.round(obj['results']))

    t = PrettyTable(['Acc', 'F1'])
    t.title = 'Intention model'
    t.add_row([acc, f1])
    print(t)

def print_pikle(path):
    obj = pd.read_pickle(path)
    print(obj)

def update_dict():
    data_opts = {'fstride': 1,
                'sample_type': 'all', 
                'height_rng': [0, float('inf')],
                'squarify_ratio': 0,
                'data_split_type': 'default',  #  kfold, random, default
                'seq_type': 'intention', #  crossing , intention
                'min_track_size': 0, #  discard tracks that are shorter
                'max_size_observe': 15,  # number of observation frames
                'max_size_predict': 5,  # number of prediction frames
                'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                'balance': True,  # balance the training and testing samples
                'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                'encoder_input_type': [],
                'decoder_input_type': ['bbox'],
                'output_type': ['intention_binary']
                }

    params = {'fstride': 1,
            'sample_type': 'all',  # 'beh'
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'default',  # kfold, random, default
            'seq_type': 'intention',
            'min_track_size': 15,
            'random_params': {'ratios': None,
                            'val_data': True,
                            'regen_data': False},
            'kfold_params': {'num_folds': 5, 'fold': 1}}
    
    params.update(data_opts)
    print(params)

with open('config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

# print("Check if running in virtual environment: ", check_venv())
print("Check if GPU is available: ", check_gpu())

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# check_images(config_file)

#print_intent_results(config_file['PRETRAINED_MODEL_PATH'])
#print_pikle(os.path.join(config_file['PRETRAINED_MODEL_PATH'], 'history.pkl'))

#update_dict()