# import os
#
# input_file = "D:/PycharmProjects/Findex/Audio/RawAudios/15-12.mp4"
# output_file = "D:/PycharmProjects/Findex/Audio/RawAudios/15-12.mp3"
#
# os.system(f"ffmpeg -i {input_file} -q:a 0 -map a {output_file}")

from pydub import AudioSegment

a = AudioSegment.from_file("/Audio/RawAudios/15-12.mp3", "mp3")
a = a[:5*60*1000]
a.export("D:/PycharmProjects/Findex/Audio/RawAudios/15-12-test.mp3", 'mp3')

