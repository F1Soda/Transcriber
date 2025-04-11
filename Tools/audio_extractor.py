import os

input_file = "D:/PycharmProjects/Findex/Audio/RawAudios/15-12.mp4"
output_file = "D:/PycharmProjects/Findex/Audio/RawAudios/15-12.mp3"

os.system(f"ffmpeg -i {input_file} -q:a 0 -map a {output_file}")