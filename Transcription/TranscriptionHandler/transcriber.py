﻿import numpy as np
import os
import torch
import soundfile as sf
import time
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from Audio.AudioHandler.preprocessor import Preprocessor

class Transcriber:
    """
    Class interface above whisper for getting transcription from preprocessed audio
    """
    # initialization
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

    @staticmethod
    def speech_to_text(audio_path, speech_timestamps, output_path: str = None):
        print(f"Run speech_to_text on {audio_path}")
        start_time = time.time()
        if not output_path:
            filename = os.path.basename(os.path.dirname(audio_path))
            output_path = os.path.join(Transcriber.save_dir, filename) + '.txt'

        waveforms, new_speech_timestamps = Transcriber._get_waveforms(audio_path, speech_timestamps)

        results = Transcriber.asr_pipeline(
            waveforms,
            generate_kwargs={"language": "russian"},
        )

        # Transcribe and save
        with open(output_path, "w", encoding="utf-8") as out_file:
            for idx, (asr, pair) in enumerate(zip(results,new_speech_timestamps)):
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

        print(f"Transcript saved to: {output_path}. Duration: {time.time() - start_time}")

    @staticmethod
    def _get_waveforms(audio_path, speech_timestamps):
        padding = 0.8 # how much seconds add to start from origin audio to each chunk
        max_duration = 30.0  # maximum length of concatenated waveform in seconds
        silence_pad = 1.0  # seconds of silence before and after each chunk
        silence_pad_in_one_chunk = 0.8  # seconds of silence before and after each chunk

        combined_waveforms = []
        combined_timestamps = []

        with sf.SoundFile(audio_path) as f:
            sample_rate = f.samplerate
            chunk = []
            chunk_duration = 0.0
            chunk_start_index = None

            for i, pair in enumerate(speech_timestamps):
                start_time = max(pair['start'] - padding, 0)
                end_time = pair['end']
                duration = end_time - start_time

                if chunk_duration + duration >= max_duration:
                    if chunk:
                        combined_waveform = np.concatenate(chunk)
                        silence = np.zeros(int(sample_rate * silence_pad))
                        combined_waveform = np.concatenate([silence, combined_waveform, silence])
                        combined_waveforms.append(combined_waveform)

                        combined_timestamps.append({
                            'start': speech_timestamps[chunk_start_index]['start'],
                            'end': speech_timestamps[i - 1]['end']
                        })
                        chunk = []
                        chunk_duration = 0.0

                # Read and pad the current segment
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                f.seek(start_frame)
                waveform = f.read(end_frame - start_frame)

                silence = np.zeros(int(sample_rate * silence_pad_in_one_chunk))
                waveform = np.concatenate([silence, waveform, silence])

                if not chunk:
                    chunk_start_index = i

                chunk.append(waveform)
                chunk_duration += duration

            # Handle the final chunk
            if chunk:
                combined_waveform = np.concatenate(chunk)
                combined_waveforms.append(combined_waveform)
                combined_timestamps.append({
                    'start': speech_timestamps[chunk_start_index]['start'],
                    'end': speech_timestamps[-1]['end']
                })

        return combined_waveforms, combined_timestamps