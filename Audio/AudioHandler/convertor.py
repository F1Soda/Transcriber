from pydub import AudioSegment, effects
import torch
import multiprocessing
from denoiser import pretrained
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path



class Convertor:
    save_folder = 'Audio/ProcessedVideos'
    den = pretrained.dns64().eval()
    chunk_duration = 30
    sample_rate = 16000
    sample_width = 2
    channels = 1

    @staticmethod
    def convert_with_denoiser(file_path: str, out_dirpath: str = None, use_threading : bool = True):
        file_name = os.path.basename(file_path)

        if out_dirpath is None:
            out_dirpath = os.path.join(Convertor.save_folder, file_name.split('.')[0])
        Path(out_dirpath).mkdir(parents=True, exist_ok=False)

        count_chunks = 10

        chunk_args = [(i, out_dirpath, file_path) for i in range(count_chunks)]

        if use_threading:
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(Convertor.process_chunk, *args) for args in chunk_args]
                for future in futures:
                    future.result()
        else:
            for i in range(0, count_chunks):
                Convertor.process_chunk(*(chunk_args[i]))


    @staticmethod
    def process_chunk(index, output_dir: str, file_path: str):
        try:
            chunk = Convertor._load_partial(file_path, start_second=index * Convertor.chunk_duration, duration=Convertor.chunk_duration)

            chunk = chunk.set_frame_rate(Convertor.sample_rate).set_channels(Convertor.channels).set_sample_width(Convertor.sample_width)
            chunk = Convertor._normalize_audio(chunk)
            chunk = effects.compress_dynamic_range(chunk)

            samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
            samples = samples.unsqueeze(0)


            with torch.no_grad():
                denoised = Convertor.den(samples)

            denoised = (denoised.squeeze().numpy() * 32768).astype("int16")
            audio = AudioSegment(
                denoised.tobytes(),
                frame_rate=Convertor.sample_rate,
                sample_width=Convertor.sample_width,
                channels=Convertor.channels
            )
            filename = os.path.basename(file_path).split('.')[0] + f"-part{index}" + '.wav'
            audio.export(os.path.join(output_dir, filename), format="wav")
            print(f"Processed chunk {index}")
        except Exception as e:
            print(f"Error processing {index}: {e}")


    @staticmethod
    def _normalize_audio(audio_segment, target_dBFS=-20.0):
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        return audio_segment.apply_gain(change_in_dBFS)

    @staticmethod
    def _load_partial(file_path, start_second, duration):
        return AudioSegment.from_file(
            file_path,
            start_second=start_second,
            duration=duration
        )


