from PIL import Image, ImageDraw
import os

directory = "png"
count = 0
for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)

        i = count % 4

        if i == 0:
            im = Image.open(f)
            draw = ImageDraw.Draw(im)
            draw.rectangle([(0, 0), (im.size[0]/2, im.size[1]/2)], fill="black", outline=None, width=1)
            im.save(filename + 'black.png', 'png')
        if i == 1:
            im = Image.open(f)
            draw = ImageDraw.Draw(im)
            draw.rectangle([(im.size[0]/2, 0), (im.size[0], im.size[1]/2)], fill="black", outline=None, width=1)
            im.save(filename + 'black.png', 'png')
        if i == 2:
            im = Image.open(f)
            draw = ImageDraw.Draw(im)
            draw.rectangle([(0, im.size[1]/2), (im.size[0]/2, im.size[1])], fill="black", outline=None, width=1)
            im.save(filename + 'black.png', 'png')
        if i == 3:
            im = Image.open(f)
            draw = ImageDraw.Draw(im)
            draw.rectangle([(im.size[0]/2, im.size[1]/2), (im.size[0], im.size[1])], fill="black", outline=None, width=1)
            im.save(filename + 'black.png', 'png')
        
        count += 1