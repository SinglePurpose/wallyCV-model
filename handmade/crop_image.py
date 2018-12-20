# This script crops the original image into smaller pieces

from tqdm import tqdm
import os
from PIL import Image


# chops the images into smaller images for training
def chop(x_div, y_div, input_file_path, output_file_path):
    # create a directory if it does not exist
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    counter = 1
    for image in tqdm(os.listdir(input_file_path)):
        if image != '.DS_Store':
            img = Image.open(input_file_path + '/' + image)
            (imageWidth, imageHeight) = img.size
            grid_x = x_div
            grid_y = y_div
            range_x = int(imageWidth / grid_x)
            range_y = int(imageWidth / grid_y)
            for x in range(range_x):
                for y in range(range_y):
                    bbox = (x * grid_x, y * grid_y, x * grid_x + grid_x, y * grid_y + grid_y)
                    slice_bit = img.crop(bbox)
                    slice_bit.save(output_file_path + '/' + str(counter) + '_' + str(x) + '_' + str(y) + '.jpg',
                                   optimize=True, bits=6)
            counter += 1


if __name__ == '__main__':
    chop(x_div=512, y_div=512,
         input_file_path='PATH_INPUT_DIR',
         output_file_path='PATH_OUTPUT_DIR')
