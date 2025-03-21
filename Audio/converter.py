from pydub import AudioSegment, effects

filepath = "castdev/Максим кастдев.mp3"
out_filepath = "castdev/Максим кастдев_mono.wav"

audio = AudioSegment.from_file(filepath)
audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)


def normalize_audio(audio_segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)


audio = normalize_audio(audio)
audio = effects.compress_dynamic_range(audio)
audio.export(out_filepath, format="wav")
