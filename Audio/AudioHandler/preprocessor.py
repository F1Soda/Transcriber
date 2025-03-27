from pydub import AudioSegment, effects
from pydub.utils import mediainfo
import torch
from denoiser import pretrained
from concurrent.futures import ThreadPoolExecutor
from pydub.silence import split_on_silence
import os
from pathlib import Path
import time
import threading
import traceback
import math

lock = threading.Lock()

class Preprocessor:
    save_folder = 'Audio/ProcessedAudios'
    den = pretrained.dns64().eval()
    chunk_duration = 30
    sample_rate = 16000
    sample_width = 2
    channels = 1
    _audio = None

    @staticmethod
    def convert_with_denoiser(file_path: str, out_dirpath: str = None, use_threading: bool = True):
        print(f"start converting: {file_path}")
        file_name = os.path.basename(file_path)

        if out_dirpath is None:
            out_dirpath = os.path.join(Preprocessor.save_folder, file_name.split('.')[0])
        Path(out_dirpath).mkdir(parents=True, exist_ok=False)

        info = mediainfo(file_path)
        duration_sec = float(info['duration'])

        Preprocessor._audio = AudioSegment.from_file(file_path)

        count_chunks = math.ceil(duration_sec / Preprocessor.chunk_duration)
        chunk_args = [(i, out_dirpath, file_path) for i in range(count_chunks)]

        start_time = time.time()

        if use_threading:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(Preprocessor.process_chunk, *args) for args in chunk_args]
                for future in futures:
                    future.result()
        else:
            for i in range(0, count_chunks):
                Preprocessor.process_chunk(*(chunk_args[i]))

        print(f"Done! Time: {time.time() - start_time}")

        Preprocessor._audio = None

    @staticmethod
    def remove_long_silences(audio_segment, silence_thresh=-45, min_silence_len=2000):
        '''
        not use it
        :param audio_segment:
        :param silence_thresh:
        :param min_silence_len:
        :return:
        '''
        audio_chunks = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=1000
        )

        audio_processed = sum(audio_chunks)
        return audio_processed

    @staticmethod
    def process_chunk(index, output_dir: str, file_path: str):
        try:
            start_time = time.time()
            chunk = Preprocessor._load_partial(start_second=index * Preprocessor.chunk_duration,
                                               duration=Preprocessor.chunk_duration)


            chunk = chunk.set_frame_rate(Preprocessor.sample_rate).set_channels(Preprocessor.channels).set_sample_width(
                Preprocessor.sample_width)

            chunk = Preprocessor._normalize_audio(chunk, target_dBFS=-30)

            chunk = effects.compress_dynamic_range(
                chunk,
                threshold=-25.0,  # start compressing quieter sounds
                ratio=3.0,        # gentle compression
                attack=10.0,      # slightly slower to avoid "pumping"
                release=100.0     # smooth fade-out of compression
            )

            samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
            samples = samples.unsqueeze(0)

            with torch.no_grad():
                denoised = Preprocessor.den(samples)

            denoised = (denoised.squeeze().numpy() * 32768).astype("int16")
            audio = AudioSegment(
                denoised.tobytes(),
                frame_rate=Preprocessor.sample_rate,
                sample_width=Preprocessor.sample_width,
                channels=Preprocessor.channels
            )

            filename = os.path.basename(file_path).split('.')[0] + f"-part{index}" + '.wav'
            with lock:
                audio.export(os.path.join(output_dir, filename), format="wav")
            end_time = time.time()

            with lock:
                print(f"Processed chunk {index}. Duration: {end_time - start_time}")
        except Exception as e:
            print(f"Error processing {index}: {e}")
            traceback.print_exc()

    @staticmethod
    def _normalize_audio(audio_segment, target_dBFS=-20.0, max_gain=20.0):
        current_dBFS = audio_segment.dBFS
        change_in_dBFS = target_dBFS - current_dBFS
        change_in_dBFS = min(change_in_dBFS, max_gain)
        return audio_segment.apply_gain(change_in_dBFS)

    @staticmethod
    def _load_partial(start_second, duration):
        start_ms = int(start_second * 1000)
        end_ms = start_ms + int(duration * 1000)

        with lock:
            chunk = Preprocessor._audio[start_ms:end_ms]

        expected_length = duration * 1000
        if len(chunk) < expected_length:
            silence = AudioSegment.silent(duration=expected_length - len(chunk))
            chunk += silence

        return chunk
