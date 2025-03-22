from convertor import Convertor
import os

def main(file_path):
    Convertor.convert_with_denoiser(file_path)


if __name__ == '__main__':
    p = r'Audio/RawVideos/20-02.mp3'
    main(p)
    # audio = Convertor._load_partial(file_path, 0, 60)
    # audio.export(r'D:/PycharmProjects/Findex/no_preprocess20-02-part0.mp3', format="wav")