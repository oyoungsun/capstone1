import json
import random
import os
from tqdm import tqdm

data_type = 'htr'  # htr, ocr
labeling_filename = 'handwriting_data_info1.json'

## Check Json File
file = json.load(open(f'./kor_dataset/aihub_data/{data_type}/{labeling_filename}'))

## Separate dataset - train, validation, test

## Separate image id - train, validation, test
train_img_ids = {}
validation_img_ids = {}
test_img_ids = {}

## Write json files
print("now")
## Make gt_xxx.txt files
data_root_path = f'./test/'
save_root_path = f'./test/'

gt_file = open(f'{save_root_path}.txt', 'w')
gt_file.write(f'test\\')
print("the end")
