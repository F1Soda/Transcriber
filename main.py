from Audio.AudioHandler.preprocessor import Preprocessor
from Audio.AudioHandler.vad import VAD
from Transcription.TranscriptionHandler.transcriber import Transcriber
import time

# audios on witch will extracting transcription
# Add only name of aduios in folder Audio/RawAudios.
# Example: lectures = ['20-02.mp3']
lectures = []

def main():
    for lecture in lectures:
        start_time = time.time()
        raw_file_path = 'Audio/RawAudios/' + lecture
        processed_file_path = Preprocessor.convert_with_denoiser(raw_file_path)
        speech_timestamps = VAD.get_voice_timestamps(processed_file_path)
        Transcriber.speech_to_text(processed_file_path, speech_timestamps)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Total duration on {lecture}: {duration:.2f} seconds")


if __name__ == "__main__":
    main()