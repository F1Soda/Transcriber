from transcriber import Transcriber
from Audio.AudioHandler.vad import VAD
from Tools.utils import make_path_abs

def main(chunks_folder, speech_timestamps):
    Transcriber.speech_to_text(chunks_folder, speech_timestamps)


if __name__ == '__main__':
    dir_chunks = make_path_abs('Audio/ProcessedAudios/15-12/processed.wav')
    speech_timestamps = VAD.load_speach_timestamps('D:/PycharmProjects/Findex/Audio/Segments/15-12.txt')
    Transcriber.load()
    main(dir_chunks, speech_timestamps)
    Transcriber.unload()