from PIL import Image
import os

directory = '../'


def save_as_png():
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        img = Image.open(f)
        img.save(filename + '.png', 'png')


save_as_png()
