import os
from pydub import AudioSegment
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from Audio.AudioHandler.preprocessor import Preprocessor

class VAD:
    """
    Class for extracting timestamps with voice in audio
    """
    min_speech_duration_ms=1000
    min_silence_duration_ms=1000
    save_folder = 'Audio/Segments'

    @staticmethod
    def get_voice_timestamps(filepath):
        """
        Using silero vad for extracting time stamps with voice
        """
        # loading
        preprocessed_audio = AudioSegment.from_file(filepath)
        samples = preprocessed_audio.get_array_of_samples()
        samples_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0  # Normalize 16-bit PCM
        samples_tensor = samples_tensor.unsqueeze(0)  # Add batch dimension: shape [1, num_samples]

        # using silero
        model = load_silero_vad()
        speech_timestamps = get_speech_timestamps(samples_tensor, model, return_seconds=True,
                                                  threshold=0.3,
                                                  min_speech_duration_ms=VAD.min_speech_duration_ms,
                                                  min_silence_duration_ms=VAD.min_silence_duration_ms)

        # saving
        filename = os.path.basename(filepath).split('.')[0]
        output_txt_path = os.path.dirname(filepath) + f"{filename}.txt"
        with open(output_txt_path, 'w') as file:
            for pair in speech_timestamps:
                file.write(f"{pair['start']} {pair['end']}\n")

        print(f"VAD results saved to: {output_txt_path}")
        return speech_timestamps

    @staticmethod
    def slice_by_segments_and_merge(segments_path, audiofile_path, output_path=None):
        """
        Slice audio by segments and then merge it. By default used for testing segmentation timestamps by silero vad
        """
        # loading timestamps
        speech_timestamps = VAD.load_speach_timestamps(segments_path)

        # loading audio
        preprocessed_audio = AudioSegment.from_file(audiofile_path)
        speech_audio = AudioSegment.empty()

        # slicing
        for i, ts in enumerate(speech_timestamps):
            start_ms = int(ts['start'] * 1000)
            end_ms = int(ts['end'] * 1000)
            segment = preprocessed_audio[start_ms:end_ms]
            speech_audio += segment
            print(f"Added speech segment {i}: {ts['start']}s - {ts['end']}s")

        # saving
        if output_path is None:
            file_name = os.path.basename(segments_path)
            output_path = Preprocessor.save_folder + f'/{file_name}/{file_name}_voice.wav'

        speech_audio.export(output_path, format='wav')

    @staticmethod
    def load_speach_timestamps(file_path):
        speech_timestamps = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                start, end = line.split()
                speech_timestamps.append({'start': float(start), 'end': float(end)})

        return speech_timestamps
