﻿import os
from pydub import AudioSegment
from Tools.utils import make_path_abs
from Tools.logger import logger_
import time
import gc
from stt_profile import *


class VAD:
    """
    Class for extracting timestamps with voice in audio
    """
    save_folder = make_path_abs('Audio/Segments')

    @staticmethod
    def get_voice_timestamps(file_path, stt_profile: BaseProfile):
        """
        Using silero vad for extracting time stamps with voice
        file_path should be absolute path
        """
        start_time = time.time()
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        logger_.info(f"start vad: {file_path}.")
        # loading
        preprocessed_audio = AudioSegment.from_file(file_path)
        samples = preprocessed_audio.get_array_of_samples()
        samples_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0  # Normalize 16-bit PCM
        samples_tensor = samples_tensor.unsqueeze(0)  # Add batch dimension: shape [1, num_samples]

        # using silero
        model = load_silero_vad()
        speech_timestamps = get_speech_timestamps(
            samples_tensor, model, return_seconds=True,
            threshold=stt_profile.get_speech_timestamps_kwargs["threshold"],
            min_speech_duration_ms=stt_profile.get_speech_timestamps_kwargs["min_speech_duration_ms"],
            min_silence_duration_ms=stt_profile.get_speech_timestamps_kwargs["min_silence_duration_ms"]
        )

        # saving
        filename = os.path.basename(os.path.dirname(file_path))
        output_txt_path = os.path.join(VAD.save_folder, f"{filename}.txt")
        with open(output_txt_path, 'w') as file:
            for pair in speech_timestamps:
                file.write(f"{pair['start']} {pair['end']}\n")

        logger_.info(f"VAD results saved to: {output_txt_path}")
        duration = time.time() - start_time
        logger_.info(f"duration of get_voice_timestamps: {duration:.2f} seconds")

        del samples_tensor
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return speech_timestamps

    @staticmethod
    def slice_by_segments_and_merge(audiofile_path, output_path=None, segments_path=None, speech_timestamps=None):
        """
        Slice audio by segments and then merge it. By default used for testing segmentation timestamps by silero vad
        segments_path, audiofile_path should be absolute path
        """

        from Audio.AudioHandler.preprocessor import Preprocessor

        if speech_timestamps is None:
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
            logger_.info(f"Added speech segment {i}: {ts['start']}s - {ts['end']}s")

        # saving
        if output_path is None:
            file_name = os.path.splitext(os.path.basename(audiofile_path))[0]
            output_path = Preprocessor.save_folder + f'/{file_name}/{file_name}_voice.wav'

        speech_audio.export(output_path, format='wav')

        logger_.info("slice_by_segments_and_merge saved merged audio to: " + output_path)

        del preprocessed_audio
        gc.collect()

    @staticmethod
    def load_speach_timestamps(file_path):
        """
        file_path should be absolute path
        """
        speech_timestamps = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                start, end = line.split()
                speech_timestamps.append({'start': float(start), 'end': float(end)})

        return speech_timestamps
