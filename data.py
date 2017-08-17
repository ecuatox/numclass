import os
from PIL import Image
import numpy as np
from net import sigmoid


def read_image(filename, resize=None):
    img = Image.open(filename)
    img = img.convert('RGB')

    if resize:
        img = img.resize(resize, Image.ANTIALIAS)

    values = []

    for j in range(img.size[1]):
        for i in range(img.size[0]):
            values.append(sum(img.getpixel((i, j))) / (3 * 255))

    return values


def load_training_data(dataset, categories, img_size):
    images = [['images/%s/learn/%s/' % (dataset, categories[i]) + a for a in
               os.listdir('images/%s/learn/%s/' % (dataset, categories[i]))] for i in range(len(categories))]
    size = min(map(len, images))
    values = [[read_image(img, img_size) for img in images[i][:size]] for i in range(len(categories))]

    input_data, output_data = [], []

    for i in range(len(categories)):
        input_data.extend(values[i])

    for i in range(len(categories)):
        output = np.zeros(len(categories))
        output[i] = 1
        output_data.extend([output for _ in range(size)])

    output_data = np.array(output_data)

    np.random.seed(0)
    np.random.shuffle(input_data)
    np.random.seed(0)
    np.random.shuffle(output_data)

    return input_data, output_data


def load_testing_data(dataset, categories, img_size):
    images = [['images/%s/test/%s/' % (dataset, categories[i]) + a for a in
               os.listdir('images/%s/test/%s/' % (dataset, categories[i]))] for i in range(len(categories))]
    values = [[read_image(img, img_size) for img in images[i]] for i in range(len(categories))]
    return values
