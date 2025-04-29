from Audio.AudioHandler.vad import VAD

audio_file_path = r"D:\PycharmProjects\Findex\Audio\ProcessedAudios\21-02-15minute\21-02-15minute.wav"

speech_timestamps = VAD.load_speach_timestamps(r"D:\PycharmProjects\Findex\Audio\Segments\21-02-15minute.txt")

VAD.slice_by_segments_and_merge(audio_file_path, speech_timestamps=speech_timestamps)