#!/opt/homebrew/bin/python3

import os, sys
import numpy
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt


if len(sys.argv) > 1:
    img_name = sys.argv[1]
else:
    img_name = "images/train/100075.jpg"

clean = Image.open(img_name).convert('RGB')
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

fig = plt.figure(figsize=(15,8))
rows = 2
columns = 4

ax1_1 = fig.add_subplot(rows, columns, 1)
ax1_1.set_title("Clean image")
ax1_1.imshow(images['clean'])

ax1_2 = fig.add_subplot(rows, columns, 2)
ax1_2.set_title("Averaged image")
ax1_2.imshow(images['averaged'])

ax1_3 = fig.add_subplot(rows, columns, 3)
ax1_3.set_title("Noisy image(5)")
ax1_3.imshow(images['noisy5'])

ax1_4 = fig.add_subplot(rows, columns, 4)
ax1_4.set_title("Noisy image(10)")
ax1_4.imshow(images['noisy10'])

ax2_1 = fig.add_subplot(rows, columns, 5)
ax2_1.set_title("Noisy image(15)")
ax2_1.imshow(images['noisy15'])

ax2_2 = fig.add_subplot(rows, columns, 6)
ax2_2.set_title("Noisy image(25)")
ax2_2.imshow(images['noisy25'])

ax2_3 = fig.add_subplot(rows, columns, 7)
ax2_3.set_title("Noisy image(35)")
ax2_3.imshow(images['noisy35'])

ax2_4 = fig.add_subplot(rows, columns, 8)
ax2_4.set_title("Noisy image(50)")
ax2_4.imshow(images['noisy50'])

plt.show()
