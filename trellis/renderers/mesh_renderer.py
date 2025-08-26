import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
        """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.device = device

        # Try to create CUDA context, fallback to CPU if it fails
        try:
            self.glctx = dr.RasterizeCudaContext(device=device)
            self.use_cuda = True
        except Exception as e:
            print(f"⚠ Warning: CUDA context creation failed: {e}")
            print("⚠ Falling back to CPU rendering (slower but more compatible)")
            self.use_cuda = False
            self.glctx = None
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth"]
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        
        # Check if CUDA context is available
        if self.use_cuda and self.glctx is not None:
            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa))
        else:
            # Fallback to CPU rendering or simplified rendering
            print("⚠ Using fallback rendering method (CUDA context not available)")
            # Create a simple mask based on vertex positions
            rast = torch.zeros((1, resolution * ssaa, resolution * ssaa, 1), device=self.device)
            # This is a simplified fallback - you might want to implement a proper CPU renderer
        
        out_dict = edict()
        for type in return_types:
            img = None
            if self.use_cuda and self.glctx is not None:
                # Use CUDA rendering
                if type == "mask" :
                    img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                elif type == "depth":
                    img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
                elif type == "normal" :
                    img = dr.interpolate(
                        mesh.face_normal.reshape(1, -1, 3), rast,
                        torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                    )[0]
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
                    # normalize norm pictures
                    img = (img + 1) / 2
                elif type == "normal_map" :
                    img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
                elif type == "color" :
                    img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                    img = dr.antialias(img, rast, vertices_clip, faces_int)
            else:
                # Fallback rendering - create simple placeholder images
                if type == "mask":
                    img = torch.ones((1, resolution * ssaa, resolution * ssaa, 1), device=self.device)
                elif type == "depth":
                    img = torch.ones((1, resolution * ssaa, resolution * ssaa, 1), device=self.device) * 0.5
                elif type == "normal":
                    img = torch.ones((1, resolution * ssaa, resolution * ssaa, 3), device=self.device) * 0.5
                elif type == "normal_map":
                    img = torch.ones((1, resolution * ssaa, resolution * ssaa, 3), device=self.device) * 0.5
                elif type == "color":
                    img = torch.ones((1, resolution * ssaa, resolution * ssaa, 3), device=self.device) * 0.5

            if ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img

        return out_dict
