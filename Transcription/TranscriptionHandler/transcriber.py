import os
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from Audio.AudioHandler.convertor import Convertor

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
    device = torch.device(device)

    model_id = "antony66/whisper-large-v3-russian"
    whisper = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        device_map="auto"
    )
    processor = WhisperProcessor.from_pretrained(model_id)

    # Create ASR pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=Convertor.chunk_duration,
        stride_length_s=0,
        batch_size=8,
        torch_dtype=torch_dtype
    )


    @staticmethod
    def speech_to_text(chunks_folder: str, output_file: str = None):
        chunk_files = sorted([
            f for f in os.listdir(chunks_folder)
            if f.endswith('.wav') and 'part' in f
        ])

        # Output file path
        if output_file:
            output_path = os.path.join(Transcriber.save_dir, output_file) + '.txt'
        else:
            output_path = os.path.join(Transcriber.save_dir, os.path.basename(chunks_folder)) + '.txt'

        # Transcribe and save
        with open(output_path, "w", encoding="utf-8") as out_file:
            for idx, filename in enumerate(chunk_files):
                audio_path = os.path.join(chunks_folder, filename)
                print(f"\nProcessing: {filename}")

                waveform, sample_rate = sf.read(audio_path)
                if sample_rate != 16000:
                    raise ValueError(f"Whisper требует 16kHz, но у файла {sample_rate}Hz. Переконвертируйте!")

                time_offset = idx * Convertor.chunk_duration

                asr = Transcriber.asr_pipeline(
                    waveform,
                    generate_kwargs={"language": "russian"},
                    return_timestamps="sentence"
                )

                # Write to file
                for chunk in asr["chunks"]:
                    if "text" not in chunk or "timestamp" not in chunk:
                        continue  # skip malformed entries
                    timestamp_sec = chunk["timestamp"][0] + time_offset
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    timestamp_str = f"[{minutes:02}:{seconds:02}]"
                    out_file.write(f"{timestamp_str} {chunk['text'].strip()}\n")

        print(f"Transcript saved to: {output_path}")