class BaseProfile:
    def __init__(self, no_skip: bool = False):
        # General
        self.chunk_duration = 30

        # Preprocessor settings
        self.normalization_kwargs = {
            "target_dBFS": -30,
            "max_gain": 20
        }
        self.compress_dynamic_range_kwargs = {
            "threshold": -25.0,
            "ratio": 3.0,
            "attack": 10.0,
            "release": 100.0
        }

        # VAD settings
        self.get_speech_timestamps_kwargs = {
            "threshold" : 0.3,
            "min_speech_duration_ms": 1000,
            "min_silence_duration_ms": 1000
        }

        # Transcriber settings
        self.batch_size = 4

        self.get_waveforms = {
            "padding": 0.8,
            "silence_pad": 1,
            "silence_pad_between_sentence": 0.8
        }

        self.make_denoise = True
        self.no_skip = no_skip


class NoDeniseProfile(BaseProfile):
    def __init__(self, no_skip):
        super().__init__(no_skip)
        self.make_denoise = False


class SkipProfile(BaseProfile):
    def __init__(self, no_skip):
        super().__init__(no_skip)
        self.make_denoise = False
        self.normalization_kwargs = None
        self.compress_dynamic_range_kwargs = None


class CustomProfile(BaseProfile):
    def __init__(self, no_skip):
        super().__init__(no_skip)
        self.make_denoise = False
        self.normalization_kwargs["target_dBFS"] = 10
        self.normalization_kwargs["max_gain"] = 20
        self.normalization_kwargs = None
        self.compress_dynamic_range_kwargs["ratio"] = 2
        self.compress_dynamic_range_kwargs["threshold"] = 10
        self.compress_dynamic_range_kwargs["attack"] = 100
        self.compress_dynamic_range_kwargs["release"] = 100
        self.no_skip = True


class CustomProfile1(BaseProfile):
    def __init__(self, no_skip):
        super().__init__(no_skip)
        self.make_denoise = False
        self.normalization_kwargs["target_dBFS"] = 10
        self.normalization_kwargs["max_gain"] = 20
        self.normalization_kwargs = None
        # self.compress_dynamic_range_kwargs = None
        self.no_skip = True


class CustomProfile2(BaseProfile):
    def __init__(self, no_skip):
        super().__init__(no_skip)
        self.make_denoise = False
        self.normalization_kwargs = None
        self.compress_dynamic_range_kwargs = None
        self.no_skip = False

        self.batch_size = 6
        self.chunk_duration = 30

        self.get_speech_timestamps_kwargs['threshold'] = 0.4
        self.get_speech_timestamps_kwargs['min_speech_duration_ms'] = 2000
        self.get_speech_timestamps_kwargs['min_silence_duration_ms'] = 750

        # self.get_waveforms = {
        #     "padding": 0.5,
        #     "silence_pad": 1.5,
        #     "silence_pad_between_sentence": 1.5
        # }

