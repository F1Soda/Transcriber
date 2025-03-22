from pydub import AudioSegment, effects
import torch
from denoiser import pretrained
import os
from pathlib import Path

CHUNK_DURATION_MS = 60000  # 1 minute
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

class Convertor:
    save_folder = 'ProcessedVideos'

    @staticmethod
    def convert_with_denoiser(file_path: str, out_dirpath: str = None):
        file_name = os.path.basename(file_path)

        if out_dirpath is None:
            out_dirpath = os.path.join(Convertor.save_folder, file_name.split('.')[0])
        Path(out_dirpath).mkdir(parents=True, exist_ok=False)

        for i in range(0, 2):
            audio_chunk = Convertor._load_partial(file_path, start_second=i * 60, duration=60)
            Convertor.process_chunk(i, audio_chunk, out_dirpath, file_name)



    @staticmethod
    def process_chunk(index, chunk, output_dir: str, origin_file_name: str):
        chunk = chunk.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(SAMPLE_WIDTH)
        chunk = Convertor._normalize_audio(chunk)
        chunk = effects.compress_dynamic_range(chunk)

        samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
        samples = samples.unsqueeze(0)

        den = pretrained.dns64().eval()
        with torch.no_grad():
            denoised = den(samples)

        denoised = (denoised.squeeze().numpy() * 32768).astype("int16")
        audio = AudioSegment(
            denoised.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=SAMPLE_WIDTH,
            channels=CHANNELS
        )
        filename = origin_file_name.split('.')[0] + f"-part{index}" + '.wav'
        if output_dir:
            audio.export(os.path.join(output_dir, filename), format="wav")
        else:
            save_dir = os.path.join(Convertor.save_folder, origin_file_name)
            audio.export(os.path.join(save_dir, filename), format="wav")


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


