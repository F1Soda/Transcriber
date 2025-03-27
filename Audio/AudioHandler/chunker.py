import collections
import webrtcvad
import numpy as np
import torchaudio
import os
import torch

# Helper to frame audio into small chunks
class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    voiced_frames = []
    segments = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time = ring_buffer[0][0].timestamp
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                segments.append({'start': start_time, 'end': end_time})
                triggered = False
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        end_time = frame.timestamp + frame.duration
        segments.append({'start': start_time, 'end': end_time})

    return segments

def convert_to_16k_mono(input_path: str, output_path: str):
    waveform, sample_rate = torchaudio.load(input_path)

    # Convert to mono (if stereo)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 16000)

def get_voiced_segments(audio_path: str, aggressiveness: int = 2):
    audio, sr = torchaudio.load(audio_path)
    assert sr == 16000, "webrtcvad only supports 16kHz"
    audio = audio.squeeze().numpy()
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_bytes = audio_int16.tobytes()

    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(30, raw_bytes, sr))
    segments = vad_collector(sr, 30, 300, vad, frames)

    # Convert to milliseconds
    return [{'start': int(s['start'] * 1000), 'end': int(s['end'] * 1000)} for s in segments]
def save_voiced_chunks(audio_path: str, segments: list, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    waveform, sample_rate = torchaudio.load(audio_path)

    for idx, seg in enumerate(segments):
        start_sample = int((seg['start'] / 1000.0) * sample_rate)
        end_sample = int((seg['end'] / 1000.0) * sample_rate)

        chunk_waveform = waveform[:, start_sample:end_sample]
        chunk_path = os.path.join(output_folder, f"part{idx:03}.wav")

        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        print(f"Saved: {chunk_path}")

def merge_chunks(chunks_folder: str, count: int = 20):
    files = sorted([f for f in os.listdir(chunks_folder) if f.endswith('.wav')])
    files = files[:count]  # Take only the first 20

    waveforms = []
    for f in files:
        waveform, sr = torchaudio.load(os.path.join(chunks_folder, f))
        waveforms.append(waveform)

    # Concatenate all chunks along time axis
    full_waveform = torch.cat(waveforms, dim=1)
    return full_waveform, sr


def save_waveform(waveform, sr, path: str):
    torchaudio.save(path, waveform, sr)
    print(f"Saved merged audio to: {path}")

# merged, sr = merge_chunks("D:/PycharmProjects/Findex/Audio/ProcessedAudios/05-03-old", count=20)

# save_waveform(merged, sr, "merged.wav")

