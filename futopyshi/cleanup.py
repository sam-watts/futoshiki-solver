import glob
import os

PATHS = [
    r'data/cropped/main_cropped/',
    r'data/cropped/row_cropped/',
    r'data/cropped/col_cropped/'
]

def remove_files():
    for path in PATHS:
        files = glob.glob(path + '*')
        for file in files:
            os.remove(file)
            
if __name__ == '__main__':
    remove_files()