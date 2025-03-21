import ffmpeg

# https://www.youtube.com/watch?v=noQGuzRqQls

video_path = "castdev/Саша кастдев.mp4"
audio_path = "castdev/Саша кастдев.mp3"

ffmpeg.input(video_path).output(audio_path, format='mp3').run()
