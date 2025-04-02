from preprocessor import Preprocessor
from pydub import AudioSegment
import os

def main(file_path):
    Preprocessor.convert_with_denoiser(file_path, use_threading=True)


if __name__ == '__main__':
    p = r'Audio/RawAudios/15-12.mp3'
    main(p)
