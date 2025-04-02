from preprocessor import Preprocessor
from utils import make_path_abs
import os

def main(file_path):
    Preprocessor.convert_with_denoiser(file_path)


if __name__ == '__main__':
    p = make_path_abs(r'Audio/RawAudios/15-12-test.mp3')
    Preprocessor.load()
    main(p)
    Preprocessor.unload()
