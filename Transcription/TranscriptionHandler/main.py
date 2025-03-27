from transcriber import Transcriber
import os
import re

def main(chunks_folder):
    Transcriber.speech_to_text(chunks_folder)


if __name__ == '__main__':
    dir_chunks = r'Audio/ProcessedAudios/20-02'
    # chunk_files = {int(re.search(r'part(\d+)', os.path.basename(f)).group(1)): f for f in os.listdir(dir_chunks)}
    # chunk_files = [value for key, value in sorted(chunk_files.items())]
    # print(chunk_files)
    main(dir_chunks)