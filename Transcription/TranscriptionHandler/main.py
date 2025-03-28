from transcriber import Transcriber
from Audio.AudioHandler.vad import VAD

def main(chunks_folder, speech_timestamps):
    Transcriber.speech_to_text(chunks_folder, speech_timestamps)


if __name__ == '__main__':
    dir_chunks = r'Audio/ProcessedAudios/20-02/merged.wav'
    speech_timestamps = VAD.load_speach_timestamps('Audio/Segments/20-02.txt')
    main(dir_chunks, speech_timestamps)