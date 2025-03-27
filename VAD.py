import os
from pydub import AudioSegment
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import re

# 1. Directory with audio files
input_dir = 'D:/PycharmProjects/Findex/Audio/ProcessedAudios/20-02'

# 2. Merge all audio files in memory using pydub
combined = AudioSegment.empty()

chunk_files = {int(re.search(r'part(\d+)', os.path.basename(f)).group(1)): f for f in os.listdir(input_dir)}
chunk_files = [value for key, value in sorted(chunk_files.items())]

for filename in chunk_files:
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        audio_path = os.path.join(input_dir, filename)
        print(f"Loading {audio_path}...")
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Ensure mono, 16kHz
        combined += audio

combined.export("D:/PycharmProjects/Findex/merged10m.wav", format="wav")
#
# 3. Convert to numpy -> torch tensor for Silero VAD
samples = combined.get_array_of_samples()
samples_tensor = torch.tensor(samples, dtype=torch.float32) / 32768.0  # Normalize 16-bit PCM
samples_tensor = samples_tensor.unsqueeze(0)  # Add batch dimension: shape [1, num_samples]

# 4. Run Silero VAD
model = load_silero_vad()
speech_timestamps = get_speech_timestamps(samples_tensor, model, return_seconds=True,
                                          threshold=0.3,
                                          min_speech_duration_ms=1000,
                                          min_silence_duration_ms=1000,)

# 5. Save results
output_txt_path = 'D:/PycharmProjects/Findex/Audio/Segments/20-02.txt'
with open(output_txt_path, 'w') as file:
    for pair in speech_timestamps:
        file.write(f"{pair['start']} {pair['end']}\n")

print(f"VAD results saved to: {output_txt_path}")
speech_timestamps = []
with open('D:/PycharmProjects/Findex/Audio/Segments/20-02.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        start, end = line.split()
        speech_timestamps.append({'start': float(start), 'end': float(end)})

speech_audio = AudioSegment.empty()

for i, ts in enumerate(speech_timestamps):
    start_ms = int(ts['start'] * 1000)
    end_ms = int(ts['end'] * 1000)
    segment = combined[start_ms:end_ms]
    speech_audio += segment
    print(f"Added speech segment {i}: {ts['start']}s - {ts['end']}s")

output_path = 'D:/PycharmProjects/Findex/no_silence_v2.wav'
#
# # --- Step 5: Export new audio without silence ---
speech_audio.export(output_path, format='wav')
# print(f"\n🎉 Cleaned audio saved to: {output_path}")
