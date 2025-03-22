from pydub import AudioSegment, effects
from pydub.utils import mediainfo
import torch
from denoiser import pretrained
from concurrent.futures import ThreadPoolExecutor
from pydub.silence import split_on_silence
import os
from pathlib import Path
import math
import time
import threading
import traceback

lock = threading.Lock()

class Convertor:
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
            out_dirpath = os.path.join(Convertor.save_folder, file_name.split('.')[0])
        Path(out_dirpath).mkdir(parents=True, exist_ok=False)

        info = mediainfo(file_path)
        duration_sec = float(info['duration'])

        Convertor._audio = AudioSegment.from_file(file_path)

        count_chunks = math.ceil(duration_sec / Convertor.chunk_duration)
        chunk_args = [(i, out_dirpath, file_path) for i in range(count_chunks)]

        start_time = time.time()

        if use_threading:
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = [executor.submit(Convertor.process_chunk, *args) for args in chunk_args]
                for future in futures:
                    future.result()
        else:
            for i in range(0, count_chunks):
                Convertor.process_chunk(*(chunk_args[i]))

        print(f"Done! Time: {time.time() - start_time}")

        Convertor._audio = None

    @staticmethod
    def remove_long_silences(audio_segment, silence_thresh=-20, min_silence_len=2000):
        audio_chunks = split_on_silence(
            audio_segment,
            min_silence_len=2000,
            silence_thresh=-45,
            keep_silence=500
        )

        audio_processed = sum(audio_chunks)
        return audio_processed

    @staticmethod
    def process_chunk(index, output_dir: str, file_path: str):
        try:
            start_time = time.time()
            chunk = Convertor._load_partial(start_second=index * Convertor.chunk_duration,
                                            duration=Convertor.chunk_duration)


            chunk = Convertor.remove_long_silences(chunk)
            if chunk.duration_seconds < 2:
                with lock:
                    print(f"Processed chunk {index}. Skipping because {chunk.duration_seconds} < 2")
                return

            #end_time = time.time()
            #with lock:
            #    print(f"\t Duration 0: {end_time - start_time}")

            #start_time = end_time
            chunk = chunk.set_frame_rate(Convertor.sample_rate).set_channels(Convertor.channels).set_sample_width(
                Convertor.sample_width)

            chunk = Convertor._normalize_audio(chunk)
            #end_time = time.time()
            #with lock:
            #    print(f"\t Duration 1: {end_time - start_time}")

            #start_time = end_time
            chunk = effects.compress_dynamic_range(chunk)
            #end_time = time.time()
            #with lock:
            #    print(f"\t Duration 2: {end_time - start_time}")

            #start_time = end_time
            samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
            samples = samples.unsqueeze(0)

            with torch.no_grad():
                denoised = Convertor.den(samples)

            #end_time = time.time()
            #with lock:
            #    print(f"\t Duration 3: {end_time - start_time}")

            # start_time = end_time
            denoised = (denoised.squeeze().numpy() * 32768).astype("int16")
            audio = AudioSegment(
                denoised.tobytes(),
                frame_rate=Convertor.sample_rate,
                sample_width=Convertor.sample_width,
                channels=Convertor.channels
            )
            filename = os.path.basename(file_path).split('.')[0] + f"-part{index}" + '.wav'
            with lock:
                audio.export(os.path.join(output_dir, filename), format="wav")
            end_time = time.time()
            # print(f"\t Duration 4: {end_time - start_time}")
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
            chunk = Convertor._audio[start_ms:end_ms]

        expected_length = duration * 1000
        if len(chunk) < expected_length:
            silence = AudioSegment.silent(duration=expected_length - len(chunk))
            chunk += silence

        return chunk
