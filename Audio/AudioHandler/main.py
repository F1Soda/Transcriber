from preprocessor import Preprocessor
from pydub import AudioSegment
import os

def main(file_path):
    Preprocessor.convert_with_denoiser(file_path, use_threading=True)


if __name__ == '__main__':
    p = r'D:/PycharmProjects/Findex/Audio/RawAudios/20-02.mp3'
    main(p)
    # audio = AudioSegment.from_file(p, start_second=0, duration=30)
    # audio = Convertor._normalize_audio(audio, target_dBFS=-60, max_gain=20)
    #audio = audio.set_frame_rate(Convertor.sample_rate).set_channels(Convertor.channels).set_sample_width(
    #    Convertor.sample_width)
    # audio.export("D:/PycharmProjects/Findex/normalized30.wav", format="wav")
    # Convertor.chunk_duration = 30
    # Convertor.process_chunk(0, output_dir="D:/PycharmProjects/Findex", file_path='Audio/RawAudios/21-02.mp3')