from pydub import AudioSegment, effects
import torch
from denoiser import pretrained

filepath = "castdev/Максим кастдев.mp3"
out_filepath = "castdev/Максим кастдев_mono.mp3"

audio = AudioSegment.from_file(filepath)
audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)


def normalize_audio(audio_segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)


audio = normalize_audio(audio)
audio = effects.compress_dynamic_range(audio)

samples = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32) / 32768.0
samples = samples.unsqueeze(0)

den = pretrained.dns64().eval()
with torch.no_grad():
    denoised_audio = den(samples)

denoised_audio = (denoised_audio.squeeze().numpy() * 32768).astype("int16")
denoised_segment = AudioSegment(
    denoised_audio.tobytes(), frame_rate=16000, sample_width=2, channels=1
)

denoised_segment.export(out_filepath, format="wav")
