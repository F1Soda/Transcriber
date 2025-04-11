import traceback
from Audio.AudioHandler.preprocessor import Preprocessor
from Audio.AudioHandler.vad import VAD
from Transcription.TranscriptionHandler.transcriber import Transcriber
from Tools.utils import make_path_abs
from Tools.logger import logger_
from multiprocessing import Process
import time
import sys
import psutil, os

def test():
    logger_.info("Run test")
    main(['Test/15-12-1.mp3','Test/15-12-2.mp3','Test/15-12-3.mp3'])

def process_lecture(lecture):
    try:
        start_time = time.time()
        raw_file_path = 'Audio/RawAudios/' + lecture
        raw_file_path = make_path_abs(raw_file_path)

        # Firstly we prepare audio by Preprocessor(For more details look inside class)
        processed_file_path = Preprocessor.handle(raw_file_path)

        # Then we applying Silero VAD for getting timestamps with voices
        speech_timestamps = VAD.get_voice_timestamps(processed_file_path)

        # After that we start transcription actions:
        # Starting logic for transcription(For more details look inside class)
        Transcriber.handle(processed_file_path, speech_timestamps)

        end_time = time.time()
        duration = end_time - start_time
        logger_.info(f"Total duration on {lecture}: {duration:.2f} seconds")
        logger_.info(f"Memory after lecture: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

    except Exception as e:
        logger_.error(f"Error: {e}. Skip {lecture}, go to next.")
        logger_.error("Traceback:\n%s", traceback.format_exc())

def main(lectures):
    for lecture in lectures:
        p = Process(target=process_lecture, args=(lecture,))
        p.start()
        p.join()  # wait for the process to finish



# audios on witch will extracting transcription. Add only name of audios in folder Audio/RawAudios.
# Example: lectures = ['20-02.mp3']
lectures = ['05-03.mp3','06-03.mp3','21-02.mp3','28-02.mp3']


if __name__ == "__main__":
    try:
        if '--test' in sys.argv:
            test()
        else:
            main(lectures)
    except Exception as e:
        logger_.error(f"Error: {e}")
        logger_.error("Traceback:\n%s", traceback.format_exc())