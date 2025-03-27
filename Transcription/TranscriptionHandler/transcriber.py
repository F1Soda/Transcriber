﻿import math
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
        stride_length_s=0,
        batch_size=8,
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

        waveforms = []
        with sf.SoundFile('D:/PycharmProjects/Findex/merged10m.wav') as f:
            for pair in speech_timestamps:
                start_time = pair['start']
                end_time = pair['end']

                sample_rate = f.samplerate
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                f.seek(start_frame)
                frames_to_read = end_frame - start_frame
                waveform = f.read(frames_to_read)
                waveforms.append(waveform)

        results = Transcriber.asr_pipeline(
            waveforms,
            generate_kwargs={"language": "russian"},
            return_timestamps="sentence"
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

                for chunk in asr["chunks"]:
                    timestamp = chunk.get("timestamp")
                    text = chunk.get("text", "").strip()

                    if not text:
                        continue
                    if timestamp is None or timestamp[0] is None:
                        timestamp = [0]

                    timestamp_sec = timestamp[0] + time_offset
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    timestamp_str = f"[{minutes:02}:{seconds:02}]"
                    chunk_number = '{:>3}'.format(idx)
                    res = f"{chunk_number} {timestamp_str} {text}\n"
                    print(res, end="")
                    out_file.write(res)

        print(f"Transcript saved to: {output_path}")