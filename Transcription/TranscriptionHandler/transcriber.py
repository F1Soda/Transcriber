import numpy as np
import os
import torch
import soundfile as sf
import time
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from Audio.AudioHandler.preprocessor import Preprocessor
from Tools.utils import make_path_abs
from Tools.logger import logger_
import gc
from stt_profile import *


class Transcriber:
    """
    Class interface above whisper for getting transcription from preprocessed audio
    """
    # initialization
    torch_dtype = torch.bfloat16
    device = 'cpu'
    save_dir = make_path_abs('Transcription/Lectures')
    asr_pipeline = None
    whisper = None
    processor = None

    @staticmethod
    def load(stt_profile: BaseProfile):
        """
        batch_size: count of parts, that will be processed on GPU together. Change it for better performance
        """
        if torch.cuda.is_available():
            Transcriber.device = 'cuda'
        elif torch.backends.mps.is_available():
            Transcriber.device = 'mps'
            setattr(torch.distributed, "is_initialized", lambda: False)

        Transcriber.torch_dtype = torch.float32  # on CPU
        if torch.cuda.is_available():
            Transcriber.torch_dtype = torch.bfloat16

        Transcriber.device = torch.device(Transcriber.device)

        model_id = "antony66/whisper-large-v3-russian"
        Transcriber.whisper = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=Transcriber.torch_dtype,
            use_safetensors=True,
            device_map="auto"
        )
        Transcriber.processor = WhisperProcessor.from_pretrained(model_id)
        # Create ASR pipeline
        Transcriber.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=Transcriber.whisper,
            tokenizer=Transcriber.processor.tokenizer,
            feature_extractor=Transcriber.processor.feature_extractor,
            chunk_length_s=stt_profile.chunk_duration,
            stride_length_s=5,
            batch_size=stt_profile.batch_size,
            torch_dtype=Transcriber.torch_dtype
        )
        logger_.info(f"Transcriber data loaded")

    @staticmethod
    def unload():
        """
        Unloads the model and pipeline to free memory.
        """
        logger_.info("Unloading Whisper model and ASR pipeline...")
        del Transcriber.asr_pipeline
        del Transcriber.processor
        del Transcriber.whisper
        Transcriber.asr_pipeline = None
        Transcriber.processor = None
        Transcriber.whisper = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger_.info(f"Transcriber data unloaded")

    @staticmethod
    def handle(audio_path, speech_timestamps, output_path: str = None, preprocessor_profile: BaseProfile = None):
        try:
            if preprocessor_profile is None:
                preprocessor_profile = BaseProfile()

            # Load Whisper
            Transcriber.load(preprocessor_profile)

            Transcriber.speech_to_text(audio_path, speech_timestamps, preprocessor_profile, output_path)
            Transcriber.unload()
        except Exception as e:
            # unload denoiser
            Transcriber.unload()
            raise e

    @staticmethod
    def speech_to_text(audio_path, speech_timestamps, stt_profile: BaseProfile, output_path: str = None):
        """
        audio_path, output_path should be absolute path
        """
        logger_.info(f"Run speech_to_text on {audio_path}")
        start_time = time.time()
        if not output_path:
            filename = os.path.basename(os.path.dirname(audio_path))
            output_path = os.path.join(Transcriber.save_dir, filename) + '.txt'

        waveforms, new_speech_timestamps, real_chunk_duration = Transcriber._get_waveforms(audio_path,
                                                                                           speech_timestamps,
                                                                                           stt_profile)

        results = []
        for waveform in waveforms:
            with torch.inference_mode():
                res = Transcriber.asr_pipeline(
                    waveform,
                    generate_kwargs={"language": "russian"},
                )
            results.append(res)
            del res
            gc.collect()
            torch.cuda.empty_cache()


        del waveforms

        # Transcribe and save
        with open(output_path, "w", encoding="utf-8") as out_file:
            for idx, (asr, pair, duration) in enumerate(zip(results, new_speech_timestamps, real_chunk_duration)):
                time_offset = pair['start']
                text = asr.get("text")
                if not text:
                    continue

                timestamp_sec = time_offset
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                timestamp_str = f"[{minutes:02}:{seconds:02}]"
                chunk_number = '{:>3}'.format(idx)
                duration = '{:>3}'.format(int(duration))
                res = f"{chunk_number} {timestamp_str} {duration} {text}\n"
                out_file.write(res)
                logger_.info(res)

        logger_.info(f"Transcript saved to: {output_path}. Duration: {time.time() - start_time}")

    @staticmethod
    def _get_waveforms(audio_path, speech_timestamps, stt_profile):
        """
        audio_path should be absolute path
        """
        # TODO: Rename all that shit to normal names and write better description
        # how much seconds capture before start chunk
        padding = stt_profile.get_waveforms["padding"]
        # maximum length of concatenated waveform in seconds
        max_duration = stt_profile.chunk_duration + 10
        # seconds of silence before and after each chunk
        silence_pad = stt_profile.get_waveforms["silence_pad"]
        # seconds of silence before and after each sentence in chunk
        silence_pad_between_sentence = stt_profile.get_waveforms["silence_pad_between_sentence"]

        combined_waveforms = []
        combined_timestamps = []
        real_chunk_duration = []

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

                        real_chunk_duration.append(chunk_duration)

                        chunk = []
                        chunk_duration = 0.0

                # Read and pad the current segment
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                f.seek(start_frame)
                waveform = f.read(end_frame - start_frame)

                silence = np.zeros(int(sample_rate * silence_pad_between_sentence))
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

                real_chunk_duration.append(chunk_duration)

        return combined_waveforms, combined_timestamps, real_chunk_duration
