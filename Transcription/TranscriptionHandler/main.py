import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from transcriber import Transcriber
from Audio.AudioHandler.vad import VAD
from stt_profile import *
from Tools.utils import make_path_abs


def main(chunks_folder, speech_timestamps, output_path, profile):
    Transcriber.handle(chunks_folder, speech_timestamps, output_path, profile)


if __name__ == '__main__':
    prof = CustomProfile2(no_skip=False)

    dir_chunks = r"D:\PycharmProjects\Findex\Audio\ProcessedAudios\21-02-15minute\21-02-15minute.wav"
    speech_timestamps = VAD.get_voice_timestamps(dir_chunks, stt_profile=prof)
    # speech_timestamps = VAD.load_speach_timestamps(r'D:\PycharmProjects\Findex\Audio\Segments\21-02-15minute.txt')
    main(dir_chunks, speech_timestamps, r"D:\PycharmProjects\Findex\Transcription\Lectures\21-02-15minute.txt",
         prof)