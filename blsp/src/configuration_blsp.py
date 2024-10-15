"""BLSP config"""

from transformers import PretrainedConfig, LlamaConfig, WhisperConfig
from transformers import logging

logger = logging.get_logger(__name__)

class BlspConfig(PretrainedConfig):
    def __init__(
        self, 
        whisper_config=None, 
        llama_config=None,
        conv_kernel_sizes="5",
        adapter_inner_dim=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")
        
        if llama_config is None:
            llama_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")
        
        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        self.whisper_config['apply_spec_augment'] = True
        self.whisper_config['mask_feature_prob'] = 0.05
        self.whisper_config['mask_feature_min_masks'] = 2
        self.llama_config = LlamaConfig(**llama_config).to_dict()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.moe = False
        self.speech_encoder = 'whisper'
        self.stage = 'only_adapter' # multi-task or chat
        self.use_fbank = False