from preprocessor import Preprocessor
from Tools.utils import make_path_abs


def main(file_path):
    Preprocessor.handle(file_path)


if __name__ == '__main__':
    p = make_path_abs(r'Audio/RawAudios/15-12-test.mp3')
    Preprocessor.load()
    main(p)
    Preprocessor.unload()
