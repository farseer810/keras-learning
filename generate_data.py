#!/usr/bin/env python
#-*- coding: utf-8 -*-
import json
import os
import random

TRAIN_DATA_PATH = '/data/ILSVRC2012/image_train'
class_list = os.listdir(TRAIN_DATA_PATH)
num_classes = 100
selected_class_list = random.sample(class_list, num_classes)

train_set = []
test_set = []
validation_set = []

for c in selected_class_list:
    classdir = os.path.join(TRAIN_DATA_PATH, c)
    image_list = os.listdir(classdir)
    random.shuffle(image_list)
    num_images = len(image_list)
    num_test_data = int(num_images * 0.1)
    num_validation_data = int(num_images * 0.1)
    num_train_data = num_images - num_test_data - num_validation_data
    train_set.extend([(i, c) for i in image_list[:num_train_data]])
    test_set.extend([(i, c) for i in image_list[num_train_data:num_train_data+num_test_data]])
    validation_set.extend([(i, c) for i in image_list[num_train_data+num_test_data:]])

random.shuffle(train_set)
random.shuffle(test_set)
random.shuffle(validation_set)

data = {
    'train_set': train_set,
    'test_set': test_set,
    'validation_set': validation_set,
    'num_classes': num_classes,
    'classes': selected_class_list,
}

with open('imagenet100.json', 'wb') as f:
    f.write(json.dumps(data, indent=True).encode('utf-8'))


