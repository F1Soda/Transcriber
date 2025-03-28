from pydub import AudioSegment, effects
from pydub.utils import mediainfo
import torch
from denoiser import pretrained
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import threading
import traceback
import math
import re
import shutil

lock = threading.Lock()

class Preprocessor:
    """
    Class for preparing audio for Whisper. Make follow:
    1) slice big audio into 30 second chunks
    2) for each chunk:
        a) set sample rate to 16 kHz
        b) set channels to 1
        c) set sample width(bytes per sample)
        d) normalizing
        e) compress dynamic range
        f) applying denoiser
    3) save chunks into folder ProcessedAudios/audio_name/parts
    4) merge chunks back
    """
    save_folder = 'Audio/ProcessedAudios'
    den = pretrained.dns64().eval()
    chunk_duration = 30
    sample_rate = 16000
    sample_width = 2
    channels = 1
    _audio = None

    @staticmethod
    def convert_with_denoiser(file_path: str, out_dirpath: str = None, use_threading: bool = True, save_parts_after_merge: bool = False):
        print(f"start converting: {file_path}")
        file_name = os.path.basename(file_path)

        if out_dirpath is None:
            out_dirpath = os.path.join(Preprocessor.save_folder, file_name.split('.')[0])
        Path(out_dirpath).mkdir(parents=True, exist_ok=False)

        Preprocessor._audio = AudioSegment.from_file(file_path)
        duration_sec = float(mediainfo(file_path)['duration'])
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

        merged_path = Preprocessor.merge(out_dirpath + '/parts', save_parts_after_merge)

        print(f"Done! Time: {time.time() - start_time}")
        Preprocessor._audio = None

        return merged_path


    @staticmethod
    def merge(parts_dir: str, save_parts_after_merge):
        combined = AudioSegment.empty()

        chunk_files = {int(re.search(r'part(\d+)', os.path.basename(f)).group(1)): f for f in os.listdir(parts_dir)}
        chunk_files = [value for key, value in sorted(chunk_files.items())]

        for filename in chunk_files:
            if filename.endswith('.wav'):
                audio_path = os.path.join(parts_dir, filename)
                audio = AudioSegment.from_file(audio_path)
                # audio = audio.set_channels(1).set_frame_rate(16000)
                combined += audio

        save_path = os.path.dirname(parts_dir) + '/merged.wav'
        combined.export(save_path, format="wav")

        if save_parts_after_merge:
            return save_path

        shutil.rmtree(parts_dir)
        return save_path


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
