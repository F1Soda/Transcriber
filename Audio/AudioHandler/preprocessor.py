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
from Tools.utils import make_path_abs
from Tools.logger import logger_
import gc
from stt_profile import BaseProfile

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
    den = None
    save_folder = make_path_abs('Audio/ProcessedAudios')
    sample_rate = 16000
    sample_width = 2
    channels = 1
    _audio = None

    @staticmethod
    def load(preprocessor_profile: BaseProfile):
        if preprocessor_profile.make_denoise:
            Preprocessor.den = pretrained.dns64().eval()
        logger_.info("Preprocessor data loaded")

    @staticmethod
    def unload():
        del Preprocessor.den
        del Preprocessor._audio
        gc.collect()
        logger_.info("Preprocessor data unloaded")


    @staticmethod
    def handle(file_path: str, out_dirpath: str = None, use_threading: bool = False,
               save_parts_after_merge: bool = False, count_threads: int = 2,
               preprocessor_profile: BaseProfile = None,
               output_file_name: str = None):
        """
        out_dirpath, file_path should have absolute path!
        Not recommended to use threading, because it's can slow down executing code(pydub with denoiser work bad together on CPU)
        """
        try:
            logger_.info(f"start converting: {file_path}. Use threading: {use_threading}")
            start_time = time.time()
            if output_file_name is None:
                output_file_name = "processed_" + os.path.basename(file_path)

            # out_dirpath should have absolute path!
            if out_dirpath is None:
                out_dirpath = os.path.join(Preprocessor.save_folder, os.path.splitext(output_file_name)[0])
                out_dirpath = make_path_abs(out_dirpath)

            if not os.path.isdir(out_dirpath):
                Path(out_dirpath).mkdir(parents=True, exist_ok=True)
            elif not preprocessor_profile.no_skip and os.path.isfile(os.path.join(out_dirpath, output_file_name)):
                logger_.info(f"found already preprocessed audio. Go to next step.")
                return os.path.join(out_dirpath, output_file_name)

            if not os.path.isdir(os.path.join(out_dirpath, 'parts')):
                Path(os.path.join(out_dirpath, 'parts')).mkdir(parents=True, exist_ok=True)

            if preprocessor_profile is None:
                preprocessor_profile = BaseProfile()

            # load denoiser
            Preprocessor.load(preprocessor_profile)
            Preprocessor.preprocess(file_path, out_dirpath, use_threading, count_threads, preprocessor_profile)

            merged_path = Preprocessor.merge(os.path.join(out_dirpath, 'parts'), save_parts_after_merge, output_file_name)

            logger_.info(f"Done convert_with_denoiser! Time: {time.time() - start_time}")

            # unload denoiser
            Preprocessor.unload()

            return merged_path
        except Exception as e:
            # unload denoiser
            Preprocessor.unload()
            raise e


    @staticmethod
    def preprocess(file_path: str, out_dirpath: str, use_threading: bool, count_threads: int,
                   preprocessor_profile: BaseProfile):
        Preprocessor._audio = AudioSegment.from_file(file_path)
        duration_sec = float(mediainfo(file_path)['duration'])
        count_chunks = math.ceil(duration_sec / preprocessor_profile.chunk_duration)
        chunk_args = [(i, out_dirpath, file_path, preprocessor_profile) for i in range(count_chunks)]
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if use_threading:
            with ThreadPoolExecutor(max_workers=count_threads) as executor:
                futures = [executor.submit(Preprocessor._process_chunk, *args) for args in chunk_args]
                for future in futures:
                    future.result()
        else:
            for i in range(0, count_chunks):
                if preprocessor_profile.no_skip or not os.path.isfile(os.path.join(out_dirpath, 'parts', f'{file_name}-part{i}.wav')):
                    Preprocessor._process_chunk(*(chunk_args[i]))
                else:
                    logger_.info(f"{i} chunk already processed. Go to next")

    @staticmethod
    def merge(parts_dir: str, save_parts_after_merge, file_name : str = 'processed.wav'):
        """
        parts_dir should have absolute path!
        """

        combined = AudioSegment.empty()

        chunk_files = {int(re.search(r'part(\d+)', os.path.basename(f)).group(1)): f for f in os.listdir(parts_dir)}
        chunk_files = [value for key, value in sorted(chunk_files.items())]

        for filename in chunk_files:
            if filename.endswith('.wav'):
                audio_path = os.path.join(parts_dir, filename)
                audio = AudioSegment.from_file(audio_path)
                combined += audio

        save_path = os.path.join(os.path.dirname(parts_dir), file_name)
        combined.export(save_path, format="wav")

        if save_parts_after_merge:
            return save_path

        shutil.rmtree(parts_dir)
        return save_path

    @staticmethod
    def _process_chunk(index, output_dir: str, file_path: str, preprocessor_profile: BaseProfile):
        """
        output_dir, file_path should have absolute path!
        """
        try:
            start_time = time.time()
            chunk = Preprocessor._load_partial(start_second=index * preprocessor_profile.chunk_duration,
                                               duration=preprocessor_profile.chunk_duration)

            chunk = chunk.set_frame_rate(Preprocessor.sample_rate).set_channels(Preprocessor.channels).set_sample_width(
                Preprocessor.sample_width)

            if preprocessor_profile.normalization_kwargs:
                chunk = Preprocessor._normalize_audio(chunk, **preprocessor_profile.normalization_kwargs)

            if preprocessor_profile.compress_dynamic_range_kwargs:
                chunk = effects.compress_dynamic_range(
                    chunk,
                    **preprocessor_profile.compress_dynamic_range_kwargs
                )

            if preprocessor_profile.make_denoise:
                samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
                samples = samples.unsqueeze(0)

                with torch.no_grad():
                    denoised = Preprocessor.den(samples)

                denoised = (denoised.squeeze().numpy() * 32768).astype("int16")
                chunk = AudioSegment(
                    denoised.tobytes(),
                    frame_rate=Preprocessor.sample_rate,
                    sample_width=Preprocessor.sample_width,
                    channels=Preprocessor.channels
                )

            filename = os.path.splitext(os.path.basename(file_path))[0] + f"-part{index}" + '.wav'
            with lock:
                save_path = os.path.join(os.path.join(output_dir, "parts"), filename)
                chunk.export(save_path, format="wav")
            end_time = time.time()

            with lock:
                logger_.info(f"Processed chunk {index}. Duration: {end_time - start_time}")
        except Exception as e:
            logger_.error(f"Error processing {index}: {e}")
            logger_.error("Traceback:\n%s", traceback.format_exc())
            raise e

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
