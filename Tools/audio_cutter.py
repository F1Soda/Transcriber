from pydub import AudioSegment

base = AudioSegment.from_file("D:/PycharmProjects/Findex/Audio/RawAudios/20-02.mp3", "mp3")

a = base[:3*60*1000]
a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/15-12-1.mp3", 'mp3')

a = base[3*60*1000:6*60*1000]
a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/15-12-2.mp3", 'mp3')

a = base[6*60*1000:9*60*1000]
a.export("D:/PycharmProjects/Findex/Audio/RawAudios/Test/15-12-3.mp3", 'mp3')