
import torch
from .typing import *

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False


class DummyRasterizerContext:
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get('device', torch.device('cpu'))

    def vertex_transform(self, verts, mvp_mtx):
        # Return input or zeros as a placeholder
        return torch.zeros((1, verts.shape[0], 4), device=verts.device)

    def rasterize(self, pos, tri, resolution):
        # Return dummy rasterization output
        B = pos.shape[0]
        return torch.zeros((B, 0, 0, 0), device=pos.device)


class NVDiffRasterizerContext:
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        if device.type == "cpu" or not NVDIFFRAST_AVAILABLE:
            self.ctx = DummyRasterizerContext(device=device)
            self.is_dummy = True
        else:
            self.ctx = self.initialize_context(context_type, device)
            self.is_dummy = False

    def initialize_context(
        self, context_type: str, device: torch.device
    ):
        return dr.RasterizeCudaContext(device=device) if context_type == "cuda" else dr.RasterizeGLContext(device=device)

    def vertex_transform(self, verts, mvp_mtx):
        if self.device.type == "cpu" or self.is_dummy:
            return DummyRasterizerContext().vertex_transform(verts, mvp_mtx)
        with torch.cuda.amp.autocast(enabled=False):
            verts_homo = torch.cat(
                [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
            )
            verts_clip = torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))
        return verts_clip

    def rasterize(self, pos, tri, resolution):
        if self.device.type == "cpu" or self.is_dummy:
            return DummyRasterizerContext().rasterize(pos, tri, resolution)
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
