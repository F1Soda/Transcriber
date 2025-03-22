from convertor import Convertor
from pydub import AudioSegment
import os

def main(file_path):
    Convertor.convert_with_denoiser(file_path, use_threading=True)


if __name__ == '__main__':
    p = r'Audio/RawAudios/05-03.mp3'
    main(p)
    # audio = AudioSegment.from_file(p, start_second=0, duration=60*5)
    # audio.export(r'D:/PycharmProjects/Findex/no_preprocess20-02-part0.wav', format="wav")