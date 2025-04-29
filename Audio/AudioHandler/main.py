from preprocessor import Preprocessor
from Tools.utils import make_path_abs
from stt_profile import *

def main(file_path, out_dirpath = None, file_name = None, preprocessor_profile= None):
    Preprocessor.handle(file_path, out_dirpath, preprocessor_profile=preprocessor_profile, output_file_name=file_name)


if __name__ == '__main__':
    p = make_path_abs(r'Audio/RawAudios/Test/21-02-15minute.mp3')
    op = r"D:\PycharmProjects\Findex\Audio\ProcessedAudios\21-02-15minute"
    file_name = "21-02-15minute.wav"
    main(p, op, file_name=file_name, preprocessor_profile=SkipProfile(no_skip=True))
