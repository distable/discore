from pathlib import Path

import cv2
import numpy as np
import torch

from src.classes.Plugin import Plugin
from src.lib import devices
from src.lib.loglib import trace_decorator
from src.party import tricks
from src.plugins import plugfun
from src.rendering.hud import hud


class Midas3DPlugin(Plugin):
    def __init__(self, dirpath: Path = None, id: str = None):
        super().__init__(dirpath, id)
        self.is_model_loaded = None

    def title(self):
        return "midas3d"

    def describe(self):
        return ""

    def load_model(self):
        if self.is_model_loaded: return
        self.is_model_loaded = True

        print("Loading midas ...")
        from .depth import DepthModel
        self.model = DepthModel(devices.device)
        self.model.download_midas(self.res())
        self.model.download_adabins(self.res())

        self.model.load_midas(self.res())
        self.model.load_adabins(self.res())

    def mat3d_dev(self, *kargs, **kwargs):
        if 'x' in kwargs: kwargs['x'] *= tricks.TRANSLATION_SCALE * 50
        if 'y' in kwargs: kwargs['y'] *= tricks.TRANSLATION_SCALE * 50
        if 'z' in kwargs: kwargs['z'] *= tricks.TRANSLATION_SCALE
        if 'rz' in kwargs: kwargs['r'] = kwargs.pop('rz')

        return tricks.mat2d(*kargs, **kwargs)

    @plugfun(mat3d_dev)
    @trace_decorator
    def mat3d(self,
              image=None,
              x: float = 0,
              y: float = 0,
              z: float = 0,
              rx: float = 0,
              ry: float = 0,
              rz: float = 0,
              fov: float = 90,
              near: float = 200,
              far: float = 10000,
              w_midas: float = 0.3,
              padding_mode: str = 'border',
              sampling_mode: str = 'bicubic',
              depth=None,
              flat: bool = False,
              **kwargs):
        from src import renderer
        import torch
        rv = renderer.rv

        img = rv.img
        if image is not None:
            img = image
        if img is None:
            return

        if flat:
            torch.ones((img.width, img.height), device=devices.device)
        if isinstance(depth, np.ndarray) and len(depth.shape) == 3:
            depth = tricks.bgr_depth_to_tensor(depth)

        if depth is None:
            self.load_model()

        hud(xyz=(x, y, z), rot=(rx, ry, rz), fov=fov, clip=(near, far), w=w_midas, pm=padding_mode, sm=sampling_mode)

        return tricks.transform_3d(img, depth, x, y, z, rx, ry, rz, fov, near, far, padding_mode, sampling_mode)


