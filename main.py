import traceback
from Audio.AudioHandler.preprocessor import Preprocessor
from Audio.AudioHandler.vad import VAD
from Transcription.TranscriptionHandler.transcriber import Transcriber
from utils import make_path_abs
from logger import logger_
import time


# audios on witch will extracting transcription. Add only name of audios in folder Audio/RawAudios.
# Example: lectures = ['20-02.mp3']
lectures = ['15-12.mp3']

def main():
    # Transcriber.load() # uncomment this if you have enough RAM. Read more in README.md
    # Preprocessor.load() # uncomment this if you have enough RAM. Read more in README.md

    # Pipeline is follow:
    for lecture in lectures:
        start_time = time.time()
        raw_file_path = 'Audio/RawAudios/' + lecture
        raw_file_path = make_path_abs(raw_file_path)


        # load denoiser
        Preprocessor.load()
        # Firstly we prepare audio by Preprocessor(For more details look inside class)
        processed_file_path = Preprocessor.convert_with_denoiser(raw_file_path)
        # unload denoiser
        Preprocessor.unload()

        # Then we applying Silero VAD for getting timestamps with voices
        speech_timestamps = VAD.get_voice_timestamps(processed_file_path)

        # After that we start transcription actions:
        # Load Whisper
        Transcriber.load(batch_size=8)
        # Starting logic for transcription(For more details look inside class)
        Transcriber.speech_to_text(processed_file_path, speech_timestamps)
        # Unload model
        Transcriber.unload()

        # I make load/unload because models take a lot of RAM. If you think, that you have enough RAM
        # for storage audio and model, you can comment above lines and uncomment lines above for loop.

        end_time = time.time()
        duration = end_time - start_time
        logger_.info(f"Total duration on {lecture}: {duration:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger_.error(f"Error: {e}")
        logger_.error("Traceback:\n%s", traceback.format_exc())