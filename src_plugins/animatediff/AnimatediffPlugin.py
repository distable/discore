from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src import plugins, renderer
from src.classes.Plugin import Plugin
from src_plugins.animatediff.AnimateDiff.animatediff.models.unet import UNet3DConditionModel
from src_plugins.animatediff.AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline

MOTION_MODULE_RES = 'mm_sd_v14.ckpt'


class AnimatediffPlugin(Plugin):
    def __init__(self, dirpath: Path = None, id: str = None):
        super().__init__(dirpath, id)
        self.initialized = False
        self.pipeline = None

    def title(self):
        return "animatediff"

    def describe(self):
        return ""

    def install(self):
        pass

    def uninstall(self):
        pass

    def init_anim(self):
        if self.initialized:
            pass
        self.initialized = True

        diffplug = plugins.get('sd_diffusers')
        pipeline_src = diffplug.ptxt

        path = hf_hub_download(diffplug.sd_model_url, subfolder='unet', filename='config.json')
        path = Path(path).parent.parent
        path = path.as_posix()

        inference_config = OmegaConf.load(self.src('AnimateDiff/configs/inference/inference.yaml'))
        unet_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)
        pipeline = AnimationPipeline(
            vae=pipeline_src.vae,
            text_encoder=pipeline_src.text_encoder,
            tokenizer=pipeline_src.tokenizer,
            unet=UNet3DConditionModel.from_pretrained_2d(path, subfolder="unet",
                                                         unet_additional_kwargs=unet_kwargs),
            scheduler=pipeline_src.scheduler
            # scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        # 1. unet ckpt
        # 1.1 motion module
        motion_module_state_dict = torch.load(self.res(MOTION_MODULE_RES), map_location="cpu")
        missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        pipeline.to("cuda")
        self.pipeline = pipeline

    def unload(self):
        pass

    # 1.2 T2I
    def push_frame(self, frame):
        pass

    def animate(self,
                images,
                prompt,
                nprompt='',
                steps=25,
                cfg=7,
                seed=-1,
                n=16,
                ctx=16):
        rv = renderer.rv

        if seed != -1:
            torch.manual_seed(seed)
        else:
            torch.seed()

        batches_result = self.pipeline(
            prompt,
            init_images=images,
            negative_prompt=nprompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=rv.w,
            height=rv.h,
            video_length=n,
            temporal_context=ctx,
            strides=1,
            overlap=5
        ).videos

        import einops
        frames = batches_result[0]
        frames = einops.rearrange(frames, 'c f h w -> f h w c')
        frames = (frames * 255).astype(np.uint8)

        return frames
        # x = ndframes[0][-1]
        # return x
