import cv2
import numpy as np
import torch

from src import installer
from src.classes.convert import load_cv2
from src.lib.devices import device
from plug_repos.flower.SD_CN_Animation.FloweR.model import FloweR
from src.lib.loglib import trace_decorator
from src.party import tricks, flow_viz
from src.plugins import plugfun, plugfun_img
from src.classes.Plugin import Plugin
from src.rendering import hud
from src_plugins.flower.flow_utils import flow_renorm, frames_norm, occl_renorm


# from FloweR.model import FloweR
# from FloweR.utils import flow_viz
# from FloweR import flow_utils


class FlowerFlowPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = -1
        self.h = -1
        self.model = None

    def title(self):
        return "flow_flower"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def ensure_loaded(self, h, w):
        pth = self.res("FloweR_0.1.pth")
        if not pth.exists():
            installer.gdown(pth, "https://drive.google.com/uc?export=download&id=1WhzoVIw6Kdg4EjfK9LaTLqFm5dF-IJ7F")

        if self.w != w or self.h != h:
            self.w = w
            self.h = h
            self.clip_frames = np.zeros((4, h, w, 3), dtype=np.uint8)
            self.model = FloweR(input_size=(h, w))
            self.model.load_state_dict(torch.load(pth))
            # Move the model to the device
            self.model = self.model.to(device)
            print("FlowerR model loaded.")

    def push(self, img):
        img = load_cv2(img)

        self.ensure_loaded(img.shape[0], img.shape[1])

        self.clip_frames = np.roll(self.clip_frames, -1, axis=0)
        self.clip_frames[-1] = img

    @plugfun(plugfun_img)
    @trace_decorator
    def flow(self, image, strength=1, flow=None):
        if image is None:
            return

        if flow is None:
            flow = self.get_flow(image, strength)

        w = image.shape[1]
        h = image.shape[0]
        ow = w
        oh = h

        new_xs = flow[:, :, 0]
        new_ys = flow[:, :, 1]
        new_xs = new_xs / (new_xs.max()) * (ow - 1)
        new_ys = new_ys / (new_ys.max()) * (oh - 1)

        warped_frame = cv2.remap(image,
                                 new_xs,
                                 new_ys,
                                 cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT_101)

        hud.snap('flower_before', image)
        hud.snap('flower_warped', warped_frame)
        return warped_frame

    def get_flow(self, image, strength, as_img=False):
        w = image.shape[1]
        h = image.shape[0]
        w = w // 128 * 128
        h = h // 128 * 128
        # w = 512
        # h = 512
        hud.hud(flower=strength)
        self.ensure_loaded(h, w)
        im = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        self.push(im)
        clip_frames_torch = frames_norm(torch.from_numpy(self.clip_frames).to(device, dtype=torch.float32))
        with torch.no_grad():
            pred_data = self.model(clip_frames_torch.unsqueeze(0))[0]
        pred_flow = flow_renorm(pred_data[..., :2]).cpu().numpy()
        pred_occl = occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)
        pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05)
        pred_flow = cv2.GaussianBlur(pred_flow, (31, 31), 1, cv2.BORDER_REFLECT_101)
        pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
        pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
        pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)
        pred_flow = pred_flow * strength
        flow_map = pred_flow.copy()
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
        flow_map = cv2.resize(flow_map,
                              (image.shape[1], image.shape[0]),
                              flow_map,
                              interpolation=cv2.INTER_CUBIC)
        flow_map = tricks.cancel_global_motion(flow_map)

        if as_img:
            # Flow image
            # frames_img = cv2.hconcat(list(self.clip_frames))
            # data_img = cv2.hconcat([flow_img, pred_occl, warped_frame])
            flow_img = flow_viz.flow_to_image(pred_flow)
            return flow_img

        hud.snap('flower_flow', flow_map)
        return flow_map
