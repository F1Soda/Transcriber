from transcriber import Transcriber
import os

def main(chunks_folder):
    Transcriber.speech_to_text(chunks_folder)


if __name__ == '__main__':
    dir_chunks = r'Audio/ProcessedVideos/20-02'
    main(dir_chunks)