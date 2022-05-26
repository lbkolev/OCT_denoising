from PIL import Image
import os, numpy

output_test_dir = 'averaged/test'
output_train_dir = 'averaged/train'

noisy_test_dir = ['noisy5/test', 'noisy10/test', 'noisy15/test', 'noisy25/test', 'noisy35/test', 'noisy50/test']
noisy_train_dir = ['noisy5/train', 'noisy10/train', 'noisy15/train', 'noisy25/train', 'noisy35/train', 'noisy50/train']


images_count = 6
for x in range(0,50):
    img_name = str(x).zfill(4) + '.png'

    w,h = Image.open(f"original/{img_name}").size
    arr=numpy.zeros((h,w,3), float)
    for noisy_dir in noisy_train_dir:
        full_img_path = f"{noisy_dir}/{img_name}"
        w,h = Image.open(full_img_path).size
        print(full_img_path)
        print((w,h))
        imarr =numpy.array(Image.open(full_img_path), dtype=float)
        arr = arr + imarr/images_count

    arr = numpy.array(numpy.round(arr),dtype=numpy.uint8)
    out = Image.fromarray(arr, mode='RGB')
    out.save(f"{output_train_dir}/{img_name}")

for x in range(50, 68):
    img_name = str(x).zfill(4) + '.png'

    w,h = Image.open(f"original/{img_name}").size
    arr=numpy.zeros((h,w,3), float)
    for noisy_dir in noisy_test_dir:
        full_img_path = f"{noisy_dir}/{img_name}"
        w,h = Image.open(full_img_path).size
        print(full_img_path)
        print((w,h))
        imarr =numpy.array(Image.open(full_img_path), dtype=float)
        arr = arr + imarr/images_count

    arr = numpy.array(numpy.round(arr),dtype=numpy.uint8)
    out = Image.fromarray(arr, mode='RGB')
    out.save(f"{output_test_dir}/{img_name}")

