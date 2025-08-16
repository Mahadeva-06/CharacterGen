import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "2D_Stage")))
print("[DEBUG] sys.path:", sys.path)

import safetensors.torch
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import *
from diffusers.models.modeling_utils import ModelMixin, load_state_dict, _load_state_dict_into_model
from diffusers.models.unets.unet_2d_blocks import *
from diffusers.utils import *
from diffusers import __version__
from tuneavideo.models.unet_mv2d_blocks import *
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from tuneavideo.models.transformer_mv2d import XFormersMVAttnProcessor, MVAttnProcessor
from tuneavideo.models.refunet import ReferenceOnlyAttnProc

print("[DEBUG] Starting import test...")
state_dict = safetensors.torch.load_file("./diffusers_cache/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/unet/diffusion_pytorch_model.safetensors", device="cpu")
print("[DEBUG] Loaded state_dict keys:", list(state_dict.keys())[:10], len(state_dict))
print("[DEBUG] Imports and safetensors load completed successfully.")
