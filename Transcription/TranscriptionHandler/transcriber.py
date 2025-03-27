import math
import numpy as np
import os
import re
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from Audio.AudioHandler.preprocessor import Preprocessor

class Transcriber:
    torch_dtype = torch.bfloat16
    device = 'cpu'
    save_dir = 'Transcription/Lectures'
    asr_pipeline = None

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        setattr(torch.distributed, "is_initialized", lambda: False)
    print("using: " + device)
    device = torch.device(device)

    model_id = "antony66/whisper-large-v3-russian"
    whisper = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(model_id)
    print(f"Model {model_id} was loaded")
    # Create ASR pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=Preprocessor.chunk_duration,
        stride_length_s=5,
        batch_size=16,
        torch_dtype=torch_dtype
    )
    print("pipeline is ready")

    @staticmethod
    def speech_to_text(chunks_folder: str, output_file: str = None):
        print(f"Run speech_to_text on folder {chunks_folder}")
        chunk_files = {int(re.search(r'part(\d+)', os.path.basename(f)).group(1)): f for f in os.listdir(chunks_folder)}
        chunk_files = [value for key, value in sorted(chunk_files.items())]

        # Output file path
        if output_file:
            output_path = os.path.join(Transcriber.save_dir, output_file) + '.txt'
        else:
            output_path = os.path.join(Transcriber.save_dir, os.path.basename(chunks_folder)) + '.txt'

        ITER_COUNT = 20

        # for k in range(math.ceil(len(chunk_files) / ITER_COUNT)):
        #     waveforms = []
        #     filenames = []
        #
        #     for u in range(ITER_COUNT*k, min(ITER_COUNT*k + ITER_COUNT, len(chunk_files))):
        #         fname = chunk_files[u]
        #         path = os.path.join(chunks_folder, fname)
        #         waveform, sample_rate = sf.read(path)
        #
        #         if sample_rate != 16000:
        #             raise ValueError(f"Whisper требует 16kHz, но у файла {sample_rate}Hz. Переконвертируйте!")
        #
        #         waveforms.append(waveform)
        #         filenames.append(fname)
        speech_timestamps = []
        with open('D:/PycharmProjects/Findex/Audio/Segments/20-02.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                start, end = line.split()
                speech_timestamps.append({'start': float(start), 'end': float(end)})
        speech_timestamps = speech_timestamps[70:100]
        # speech_timestamps = [{'start': 1106.2, 'end': 1116.0}, {'start': 1117.1, 'end': 1123.5}]
        padding = 0.8  # seconds of padding before and after speech

        waveforms = []

        with sf.SoundFile('D:/PycharmProjects/Findex/merged10m.wav') as f:
            sample_rate = f.samplerate
            total_frames = len(f)

            for pair in speech_timestamps:
                start_time = max(pair['start'] - padding, 0)
                end_time = pair['end'] # min(pair['end'] + padding, total_frames / sample_rate)

                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                f.seek(start_frame)
                frames_to_read = end_frame - start_frame
                waveform = f.read(frames_to_read)
                silence = np.zeros(int(sample_rate * 1.0))  # 1 second of silence
                waveform = np.concatenate([silence, waveform, silence])
                waveforms.append(waveform)

        results = Transcriber.asr_pipeline(
            waveforms,
            generate_kwargs={"language": "russian"},
        )

        # Transcribe and save
        with open(output_path, "w", encoding="utf-8") as out_file:
            for idx, (asr, pair) in enumerate(zip(results,speech_timestamps)):
                #audio_path = os.path.join(chunks_folder, filename)

                #correct_idx = int(re.search(r'part(\d+)', os.path.basename(filename)).group(1))

                #print(f"\nProcessing: {filename}")

                #waveform, sample_rate = sf.read(audio_path)
                #if sample_rate != 16000:
                #    raise ValueError(f"Whisper требует 16kHz, но у файла {sample_rate}Hz. Переконвертируйте!")

                time_offset = pair['start']
                text = asr.get("text")
                if not text:
                    continue

                timestamp_sec = time_offset
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                timestamp_str = f"[{minutes:02}:{seconds:02}]"
                chunk_number = '{:>3}'.format(idx)
                res = f"{chunk_number} {timestamp_str} {text}\n"
                print(res, end="")
                out_file.write(res)

        print(f"Transcript saved to: {output_path}")