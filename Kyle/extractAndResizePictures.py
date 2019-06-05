import sys, os
from PIL import Image
import tarfile
import shutil

    
def unpack_tar(tar_path, temp_dir):
    tar = tarfile.open(tar_path)
    tar.extractall(temp_dir)
    tar.close()
    print('Done unpacking %s' % tar_path)
    
def resize_and_move(temp_dir, dest_dir):
    moved = 0
    for subdir, dirs, files in os.walk(temp_dir):
        for file in files:
            if file[-3] in ('j','J'): 
                img_path = os.path.join(subdir, file)
                dest_path = os.path.join(dest_dir, file)

                img = Image.open(img_path)
                w,h = img.size
                mindim = min(w,h)
                size = (int(w*224/mindim), int(h*224/mindim))

                img = img.resize(size, resample = Image.BILINEAR)
                img.save(dest_path)
                
                os.remove(img_path)
                moved += 1
    return moved
        
def cleanup():
    for file in os.listdir(tar_dir):
        file_path = os.path.join(tar_dir, file)
        #if os.path.isfile(file_path):
        #    os.remove(file_path)
        if os.path.isdir(file_path): 
            shutil.rmtree(file_path)
            
def main():
    if len(sys.argv) < 3:
        print('extractAndResizePictures.py start end, to process tars [start, end)')
        return
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    dest_dir = '/home/kylecshan/data/images224/all/'
    tar_dir = '/home/kylecshan/data/raw/'
    
    for i in range(start, end):
        tar_path = tar_dir + 'images_' + str(i).zfill(3) + '.tar'
        temp_dir = '/home/kylecshan/data/raw/' + 'temp_' + str(i).zfill(3)
        os.mkdir(temp_dir)
        
        unpack_tar(tar_path, temp_dir)
        resize_and_move(temp_dir, dest_dir)
        shutil.rmtree(temp_dir)
        print('Done with tar %d' % i)

if __name__ == "__main__":
    main()