import torch
from controlnet_aux import HEDdetector
from numpy import ndarray
from PIL import Image

# from ComfyUI.custom_nodes.comfyui_controlnet_aux.src.controlnet_aux.hed import HEDdetector
from src.classes.convert import load_pil, pil2cv
from src.classes.Plugin import Plugin
from src.lib import devices

class HeddetPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heddet = None

    def title(self):
        return "heddet"

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

    def img_to_hed(self, img):
        img = load_pil(img)
        if self.heddet is None:
            # from controlnet_aux import HEDdetector

            print("Loading HED detector")
            self.heddet = HEDdetector.from_pretrained("lllyasviel/ControlNet")
            # self.heddet.netNetwork = torch.compile(self.heddet.netNetwork)
            self.heddet.netNetwork = self.heddet.netNetwork.to(devices.device)  # .netNetwork.to(device)

        if isinstance(img, ndarray):
            img = Image.fromarray(img)
        ret = self.heddet(img)
        return pil2cv(ret)
