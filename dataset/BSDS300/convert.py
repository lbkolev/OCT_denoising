#!/opt/homebrew/bin/python3

import os, sys
import numpy
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt

test_images_file = open('iids_test.txt')
test_images = [y[:-1] for y in test_images_file.readlines()]

train_images_file = open('iids_train.txt')
train_images = [y[:-1] for y in train_images_file.readlines()]

#print(test_images)
#print(train_images)

for test_image in test_images:
    full_path = f"images/test/{test_image}.jpg"
    save_full_path = f"images_averaged/test/{test_image}.jpg"

    clean = Image.open(full_path).convert('RGB')
    h, w = clean.size

    transform = tv.transforms.Compose([
        # convert it to a tensor
        tv.transforms.ToTensor(),
        # normalize it to the range [−1, 1]
        #tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    #transform_back = tv.transforms.ToPILImage()
    transform_back = tv.transforms.Compose([
        #tv.transforms.Normalize((127,127,127), (127,127,127)),
        tv.transforms.ToPILImage()
    ])

    clean = transform(clean)

    noisy5 = clean + 1/255 * 5 * torch.randn(clean.shape)
    noisy10 = clean + 1/255 * 10 * torch.randn(clean.shape)
    noisy15 = clean + 1/255 * 15 * torch.randn(clean.shape)
    noisy25 = clean + 1/255 * 25 * torch.randn(clean.shape)
    noisy35 = clean + 1/255 * 35 * torch.randn(clean.shape)
    noisy50 = clean + 1/255 * 50 * torch.randn(clean.shape)

    images = {
        'noisy5': noisy5,
        'noisy10' : noisy10,
        'noisy15' : noisy15,
        'noisy25' : noisy25,
        'noisy35' : noisy35,
        'noisy50' : noisy50,
    }

    #initialize empty tensor
    averaged = numpy.zeros((w,h,3), float)

    for image in images:
        images[image] = transform_back(images[image])
        averaged = averaged + numpy.array(images[image], dtype=float)/7
        #v.show()


    averaged = numpy.array(numpy.round(averaged),dtype=numpy.uint8)
    averaged = Image.fromarray(averaged, mode='RGB')

    clean = transform_back(clean)
    images['clean'] = clean
    images['averaged'] = averaged
   
    print(f"Image: {full_path}\nSave path: {save_full_path}")
    images['averaged'].save(save_full_path)


for train_image in train_images:
    full_path = f"images/train/{train_image}.jpg"
    save_full_path = f"images_averaged/train/{train_image}.jpg"

    clean = Image.open(full_path).convert('RGB')
    h, w = clean.size

    transform = tv.transforms.Compose([
        # convert it to a tensor
        tv.transforms.ToTensor(),
        # normalize it to the range [−1, 1]
        #tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    #transform_back = tv.transforms.ToPILImage()
    transform_back = tv.transforms.Compose([
        #tv.transforms.Normalize((127,127,127), (127,127,127)),
        tv.transforms.ToPILImage()
    ])

    clean = transform(clean)

    noisy5 = clean + 1/255 * 5 * torch.randn(clean.shape)
    noisy10 = clean + 1/255 * 10 * torch.randn(clean.shape)
    noisy15 = clean + 1/255 * 15 * torch.randn(clean.shape)
    noisy25 = clean + 1/255 * 25 * torch.randn(clean.shape)
    noisy35 = clean + 1/255 * 35 * torch.randn(clean.shape)
    noisy50 = clean + 1/255 * 50 * torch.randn(clean.shape)

    images = {
        'noisy5': noisy5,
        'noisy10' : noisy10,
        'noisy15' : noisy15,
        'noisy25' : noisy25,
        'noisy35' : noisy35,
        'noisy50' : noisy50,
    }

    #initialize empty tensor
    averaged = numpy.zeros((w,h,3), float)

    for image in images:
        images[image] = transform_back(images[image])
        averaged = averaged + numpy.array(images[image], dtype=float)/7
        #v.show()


    averaged = numpy.array(numpy.round(averaged),dtype=numpy.uint8)
    averaged = Image.fromarray(averaged, mode='RGB')

    clean = transform_back(clean)
    images['clean'] = clean
    images['averaged'] = averaged
   
    print(f"Image: {full_path}\nSave path: {save_full_path}")
    images['averaged'].save(save_full_path)
