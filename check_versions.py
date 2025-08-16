import importlib
pkgs = [
    "gradio", "rm_anime_bg", "accelerate", "transformers", "diffusers", "huggingface_hub", "safetensors", "ipdb",
    "einops", "omegaconf", "numpy", "imageio", "onnxruntime", "pytorch_lightning", "jaxtyping", "wandb", "lpips",
    "ninja", "open3d", "trimesh", "pymeshlab", "pygltflib"
]
for pkg in pkgs:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "no __version__")
        print(f"{pkg}: {v}")
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED ({e})")