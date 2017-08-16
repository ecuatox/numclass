from PIL import Image

def read_image(filename, resize=None):
    img = Image.open(filename)
    img = img.convert('RGB')

    if resize:
        img = img.resize(resize, Image.ANTIALIAS)

    values = []

    for j in range(img.size[1]):
        for i in range(img.size[0]):
            values.append(sum(img.getpixel((i, j))) / (3*255))

    return values
