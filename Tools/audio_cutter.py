# from pydub import AudioSegment
#
# base = AudioSegment.from_file("D:/PycharmProjects/Findex/Audio/RawAudios/21-02.mp3", "mp3")
#
# a = base[:1*60*1000]
# a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/21-02-1minute.mp3", 'mp3')

# a = base[3*60*1000:6*60*1000]
# a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/15-12-2.mp3", 'mp3')
#
# a = base[6*60*1000:9*60*1000]
# a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/15-12-3.mp3", 'mp3')


import subprocess

input_file = "D:/PycharmProjects/Findex/Audio/RawAudios/21-02.mp3"
output_file = "D:/PycharmProjects/Findex/Audio/RawAudios/Test/21-02-15minute.mp3"

start_time = 0  # start at 0 seconds
duration = 60 * 15   # duration in seconds

subprocess.run([
    'ffmpeg', '-ss', str(start_time),
    '-t', str(duration),
    '-i', input_file,
    '-c', 'copy',  # very important: copy mode, no re-encoding
    output_file
])
