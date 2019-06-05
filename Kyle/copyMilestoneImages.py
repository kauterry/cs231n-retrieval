import os
import shutil
from PIL import Image

def main():
    source = "/home/kylecshan/data/images/train/"
    dest = "/home/kylecshan/data/images/train_ms/"

    i = 0
    j = 0
    for file in os.scandir(source):
        i += 1
        im = Image.open(file.path)
        w,h = im.size

        if w >= 256 and h >= 256:
            shutil.copy(file.path, dest)
            j += 1
        if i % 1000 == 0:
            print("Looked at %d images, copied %d" % (i,j))
        if j >= 100000:
            break
    print("Stopped at i = %d, j = %d" % (i,j))

if __name__ == "__main__":
    main()