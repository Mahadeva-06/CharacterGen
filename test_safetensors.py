import safetensors.torch
state_dict = safetensors.torch.load_file("./diffusers_cache/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/unet/diffusion_pytorch_model.safetensors", device="cpu")
print(list(state_dict.keys())[:10], len(state_dict))
