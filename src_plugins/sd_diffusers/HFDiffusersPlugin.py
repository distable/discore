import gc
from pathlib import Path

import numpy as np
import tomesd
import torch
from compel import Compel
from transformers import CLIPTextModel

# from diffusers.utils import pt_to_pil

from jargs import args
from src import installer
from src.classes import paths
from src.classes.convert import load_cv2, load_pil
from src.lib.devices import device
from src.lib.printlib import trace_decorator, trace_decorator_noarg
from src.party import maths, tricks
from src.plugins import plugfun, plugfun_img, plugfun_redirect, plugfun_self
from src.renderer import rv
from src.rendering.hud import hud
from src.classes.Plugin import Plugin

from diffusers import AutoencoderKL, ControlNetModel, DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, UniPCMultistepScheduler

DEFAULT_HF_MODEL_PATH = 'runwayml/stable-diffusion-v1-5'
DEFAULT_HF_VAE_URL = "stabilityai/sd-vae-ft-mse"

HF_TEMPORAL_MODEL_URL = "CiaraRowles/TemporalNet"
HF_CANNY_MODEL_URL = "lllyasviel/control_v11p_sd15_softedge"
HF_HED_MODEL_URL = "lllyasviel/control_v11p_sd15_softedge"
HF_DEPTH_MODEL_URL = "lllyasviel/control_v11f1p_sd15_depth"
HF_SEG_MODEL_URL = "lllyasviel/control_v11p_sd15_seg"
HF_SHUFFLE_MODEL_URL = "lllyasviel/control_v11e_sd15_shuffle"

DEFAULT_HF_SAI_UPSCALER = "stabilityai/stable-diffusion-x4-upscaler"
DEFAULT_HF_FLOYD_IMG2IMG_URL = "DeepFloyd/IF-II-L-v1.0"
DEFAULT_HF_FLOYD_URL = "DeepFloyd/IF-I-IF-v1.0"

SD_NUM_TEXT_LAYERS = 12

TORCH_COMPILE_ENABLE = False
TORCH_COMPILE_MODE = 'reduce-overhead' # max-autotune, reduce-overhead
TOMESD_RATIO = 0
TOMESD_RATIO_CN = 0.0
# TOMESD_RATIO = .375


class HFDiffusersPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_model_url = None
        self.ptxt = None
        self.pimg = None
        self.pvar = None
        self.pvd = None
        self.pcontrolnet = None
        self.pcontrolnet_img = None
        self.pcontrolnet_inpaint = None
        self.controlnets = {}
        self.latents = None
        self.cn_names = [''] * 3
        self.last_ccg_len = 1

    def title(self):
        return "HF Diffusers"

    def load(self):
        print("Loading HF Diffusers ...")

        # self.pvar = StableDiffusionImageVariationPipeline.from_pretrained(
        #         "lambdalabs/sd-image-variations-diffusers",
        #         revision="v2.0",
        #         safety_checker=None,
        #         requires_safety_checker=Faintricate lse)
        # self.pvar.scheduler = UniPCMultistepScheduler.from_config(self.pvar.scheduler.config)
        # self.pvar.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
        # self.pvar.enable_model_cpu_offload()
        # self.pvar.enable_attention_slicing()


    def init_versatile(self):
        if self.pvd is not None: return
        print("Diffusers: Loading Versatile Diffusion...")
        from diffusers import VersatileDiffusionPipeline
        self.pvd = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion",
                                                              safety_checker=None,
                                                              requires_safety_checker=False)
        self.pvd.scheduler = UniPCMultistepScheduler.from_config(self.pvd.scheduler.config)
        self.pvd.scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000)
        self.pvd.enable_model_cpu_offload()
        self.pvd.enable_attention_slicing(1)

    @plugfun(plugfun_self('sd_diffusers'))
    def sd(self, model_url=None, clip_skip=0):
        self.init_sd(model_url, clip_skip=clip_skip)
        return self

    @plugfun(plugfun_self('sd_diffusers'))
    def cn(self, *models, sd_model_url=None, clip_skip=0):
        self.init_controlnet(models, sd_model_url=sd_model_url, clip_skip=clip_skip)
        return self

    @plugfun(plugfun_self('sd_diffusers'))
    def dfif(self):
        self.init_if()
        return self

    @plugfun(plugfun_self('sd_diffusers'))
    def lora(self, download_url):
        # # NOTE: this is for civitai only, get the download link from the actual download button
        name = self.get_civit_name(download_url)
        path = self.res(f"{name}.safetensors")
        # path = paths.hide(path)

        # installer.wget(path, download_url)
        if not path.exists():
            import os
            os.system(f"wget {download_url} -O {path}")

        print(f"Diffusers: Loading LoRA({path}) ...")
        # if self.ptxt is not None: self.ptxt.load_lora_weights(path.parent.as_posix(), weight_name=path.name)
        # if self.pcontrolnet is not None: self.pcontrolnet.load_lora_weights(path.parent.as_posix(), weight_name=path.name)
        if self.ptxt is not None: self.ptxt.load_lora_weights(str(path))
        if self.pcontrolnet is not None: self.pcontrolnet.load_lora_weights(str(path))

        return self

    def get_civit_name(self, download_url):
        name = download_url.split('/')
        name = [token for token in name if token]
        name = name[-1]
        return name

    def set_scheduler(self, scheduler):
        print(f"Diffusers: setting scheduler to {scheduler}")
        if self.pcontrolnet is not None: self.pcontrolnet.scheduler = scheduler
        if self.pcontrolnet_img is not None: self.pcontrolnet_img.scheduler = scheduler
        if self.ptxt is not None: self.ptxt.scheduler = scheduler
        if self.pimg is not None: self.pimg.scheduler = scheduler

    @plugfun()
    @trace_decorator
    def init_controlnet(self, models, sd_model_url=None, clip_skip=0):
        import torch
        def load_model(name):
            print(f"Diffusers: Loading {name} ControlNet...")
            if name == "canny": return ControlNetModel.from_pretrained(HF_CANNY_MODEL_URL, file='', torch_dtype=torch.float16)
            elif name == "hed":  return ControlNetModel.from_pretrained(HF_HED_MODEL_URL, file='', torch_dtype=torch.float16)
            elif name == "depth":  return ControlNetModel.from_pretrained(HF_DEPTH_MODEL_URL, file='', torch_dtype=torch.float16)
            elif name == "seg":  return ControlNetModel.from_pretrained(HF_SEG_MODEL_URL, file='', torch_dtype=torch.float16)
            elif name == "shuffle":  return ControlNetModel.from_pretrained(HF_SHUFFLE_MODEL_URL, file='', torch_dtype=torch.float16)
            elif name == "temporal":  return ControlNetModel.from_pretrained(HF_TEMPORAL_MODEL_URL, file='', torch_dtype=torch.float16)

        model_list = []
        for model in models:
            if model not in self.controlnets:
                self.controlnets[model] = load_model(model)
            model_list.append(self.controlnets[model])

        self.cn_names = models

        sd_model_url = self.sd_model_url or sd_model_url or DEFAULT_HF_MODEL_PATH

        # vae = AutoencoderKL.from_pretrained(sd_model_url, torch_dtype=torch.float16)
        print(f"Diffusers: Loading ControlNet with {sd_model_url}...")
        text_encoder = CLIPTextModel.from_pretrained(sd_model_url, subfolder="text_encoder", num_hidden_layers=12 - clip_skip, torch_dtype=torch.float16)
        self.pcontrolnet = StableDiffusionControlNetPipeline.from_pretrained(
                sd_model_url,
                text_encoder=text_encoder,
                safety_checker=None,
                requires_safety_checker=False,
                controlnet=model_list,
                torch_dtype=torch.float16,
                variant='fp16'
        ).to("cuda")

        self.pcontrolnet_img = StableDiffusionControlNetImg2ImgPipeline(
                vae=self.pcontrolnet.vae,
                text_encoder=self.pcontrolnet.text_encoder,
                tokenizer=self.pcontrolnet.tokenizer,
                unet=self.pcontrolnet.unet,
                scheduler=self.pcontrolnet.scheduler,
                controlnet=self.pcontrolnet.controlnet,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
        )

        if TOMESD_RATIO_CN > maths.epsilon:
            print(f"Diffusers: Applying tomesd patch")
            tomesd.apply_patch(self.pcontrolnet.unet, ratio=TOMESD_RATIO_CN, sx=2, sy=2, max_downsample=1)

        self.compel = Compel(tokenizer=self.pcontrolnet.tokenizer, text_encoder=self.pcontrolnet.text_encoder)

        if TORCH_COMPILE_ENABLE:
            self.pcontrolnet.unet.to(memory_format=torch.channels_last)
            self.pcontrolnet.controlnet.to(memory_format=torch.channels_last)

            print("Diffusers.init_controlnet: compiling unet ...")
            self.pcontrolnet.unet = torch.compile(self.pcontrolnet.unet, mode=TORCH_COMPILE_MODE, fullgraph=True)
            print("Diffusers.init_controlnet: compiling controlnet ...")
            self.pcontrolnet.controlnet = torch.compile(self.pcontrolnet.controlnet, mode=TORCH_COMPILE_MODE, fullgraph=True)

        if args.opt:
            print(f"Diffusers: enabling CPU offload")
            self.pcontrolnet.to("cpu")
            self.pcontrolnet.enable_model_cpu_offload()

    @plugfun()
    @trace_decorator
    def init_sd(self, model_url=None, clip_skip=0):
        import torch
        if self.ptxt is not None: return

        if model_url is None:
            model_url = DEFAULT_HF_MODEL_PATH
        # if 'civit' in model_url:
        #     model_path = self.res(f"{self.get_civit_name(model_url)}.safetensors")
        #     installer.wget(model_path, model_url)
        #     model_url = model_path

        self.sd_model_url = model_url

        if not self.pcontrolnet:
            print(f"Diffusers: Loading SD with {model_url}...")
            # vae = AutoencoderKL.from_pretrained(model_url, torch_dtype=torch.float16)
            text_encoder = CLIPTextModel.from_pretrained(model_url, subfolder="text_encoder", num_hidden_layers=12 - clip_skip, torch_dtype=torch.float16)
            self.ptxt = StableDiffusionPipeline.from_pretrained(model_url,
                                                                text_encoder=text_encoder,
                                                                # vae=vae,
                                                                safety_checker=None,
                                                                requires_safety_checker=False,
                                                                torch_dtype=torch.float16,
                                                                variant='fp16'
                                                                ).to("cuda")

            if args.opt:
                self.ptxt.to("cpu")
                self.ptxt.enable_model_cpu_offload()
        else:
            print("Diffusers: Loading SD from CN...")
            self.ptxt = StableDiffusionPipeline(
                    text_encoder=self.pcontrolnet.text_encoder,
                    tokenizer=self.pcontrolnet.tokenizer,
                    unet=self.pcontrolnet.unet,
                    scheduler=self.pcontrolnet.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False).to("cuda").half()

        self.pimg = StableDiffusionImg2ImgPipeline(
                vae=self.ptxt.vae,
                text_encoder=self.ptxt.text_encoder,
                tokenizer=self.ptxt.tokenizer,
                unet=self.ptxt.unet,
                scheduler=self.ptxt.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False).to("cuda")

        if TOMESD_RATIO > maths.epsilon:
            print(f"Diffusers: Applying tomesd patch")
            tomesd.apply_patch(self.ptxt.unet, ratio=TOMESD_RATIO, sx=2, sy=2, max_downsample=1)

        if self.pcontrolnet:
            self.pimg.scheduler = self.pcontrolnet.scheduler
        else:
            self.compel = Compel(tokenizer=self.ptxt.tokenizer, text_encoder=self.ptxt.text_encoder)

        if TORCH_COMPILE_ENABLE:
            self.ptxt.unet.to(memory_format=torch.channels_last)
            print("Diffusers.init_sd: compiling unet ...")
            self.ptxt.unet = torch.compile(self.ptxt.unet, mode=TORCH_COMPILE_MODE, fullgraph=True)


    @trace_decorator
    def init_if(self):
        from diffusers import IFImg2ImgPipeline
        from diffusers import IFImg2ImgSuperResolutionPipeline

        self.pif1 = IFImg2ImgPipeline.from_pretrained(DEFAULT_HF_FLOYD_URL, variant="fp16", torch_dtype=torch.float16)
        self.pif1.enable_model_cpu_offload()

        self.pif2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
                DEFAULT_HF_FLOYD_IMG2IMG_URL, text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        self.pif2.enable_model_cpu_offload()

        safety_modules = {
            "feature_extractor": self.pif1.feature_extractor,
            "safety_checker"   : self.pif1.safety_checker,
            "watermarker"      : self.pif1.watermarker,
        }
        self.pif3 = DiffusionPipeline.from_pretrained(
                DEFAULT_HF_SAI_UPSCALER, **safety_modules, torch_dtype=torch.float16
        )
        self.pif3.enable_model_cpu_offload()

    def get_dimensions(self, w, h, image):
        if w is not None and h is not None:
            return w, h
        elif isinstance(image, np.ndarray):
            return image.shape[1], image.shape[0]
        elif isinstance(image, (tuple, list)):
            if isinstance(image[0], np.ndarray):
                return image[0].shape[1], image[0].shape[0]
            else:
                raise f"Diffusers: get_dimensions: unknown image type {type(image[0])}"
        else:
            raise f"Diffusers: get_dimensions: unknown image type {type(image)}"

    def get_images(self, image):
        if isinstance(image, tuple):
            image = list(image)

        if isinstance(image, np.ndarray):
            image = load_pil(image)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = [load_pil(i) for i in image]

        return image

    @trace_decorator
    @plugfun(plugfun_redirect('sd_diffusers', 'txt2img_dev'))
    def txt2img(self,
                prompt: str = None,
                promptneg: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                **kwargs):
        self.init_sd()

        if isinstance(ccg, (tuple, list)):
            self.last_ccg_len = len(ccg)
        else:
            ccg = [ccg] * self.last_ccg_len


        hud(chg=chg, cfg=cfg, seed=seed)
        hud(**{f'ccg{i}': ccg[i-1] for i in range(1, self.last_ccg_len + 1)})
        hud(prompt=prompt)

        generator = self.get_generator(seed)
        text_embed = self.compel.build_conditioning_tensor(prompt)
        text_embed_neg = self.compel.build_conditioning_tensor(promptneg or '')

        images = self.ptxt(
                prompt_embeds=text_embed,
                negative_prompt_embeds=text_embed_neg,
                image=image,
                guidance_scale=cfg,
                output_type='np',
                num_inference_steps=int(steps),
                generator=generator).images

        image = images[0]
        image = (image * 255).round().astype("uint8")

        text_embed = None
        text_embed_neg = None
        gc.collect()

        return image

    @trace_decorator
    @plugfun(plugfun_redirect('sd_diffusers', 'img2img_dev'))
    def img2img(self,
                prompt: str = None,
                promptneg: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                w: int = 512,
                h: int = 512,
                **kwargs):
        self.init_sd()
        # return self.txt2img(self, job)
        import torch
        from src import renderer

        w, h = self.get_dimensions(w, h, image)
        image = self.get_images(image)

        if isinstance(ccg, (tuple, list)):
            self.last_ccg_len = len(ccg)
        else:
            ccg = [ccg] * self.last_ccg_len

        hud(chg=chg, cfg=cfg, seed=seed)
        hud(**{f'ccg{i}': ccg[i-1] for i in range(1, self.last_ccg_len + 1)})
        hud(prompt=prompt)


        text_embed = self.compel.build_conditioning_tensor(prompt)
        text_embed_neg = self.compel.build_conditioning_tensor(promptneg or '')
        generator = self.get_generator(seed)
        images = self.pimg(
                prompt_embeds=text_embed,
                negative_prompt_embeds=text_embed_neg,
                image=image,
                strength=maths.clamp(chg, 0, 1),
                guidance_scale=cfg,
                output_type='np',
                num_inference_steps=int(steps),
                generator=generator).images

        image = images[0]
        image = (image * 255).round().astype("uint8")

        text_embed = None
        text_embed_neg = None
        gc.collect()

        return image

    def get_generator(self, seed):
        return torch.Generator(device="cpu").manual_seed(int(seed))

    @plugfun(plugfun_redirect('sd_diffusers', 'txt2img_cn_dev'))
    @trace_decorator
    @trace_decorator_noarg
    def txt2img_cn(self,
                   prompt: str = None,
                   promptneg: str = None,
                   cfg: float = 7.0,
                   image: any = None,
                   ccg: float = 1.0,
                   chg: float = 0.01,
                   steps: int = 30,
                   seed: int = 0,
                   w: int = 512,
                   h: int = 512,
                   seed_grain: float = 0,
                   **kwargs):
        from src import renderer
        session = renderer.session

        if isinstance(ccg, (tuple, list)):
            self.last_ccg_len = len(ccg)
        else:
            ccg = [ccg] * self.last_ccg_len

        hud(chg=chg, cfg=cfg, seed=seed)
        hud(**{f'ccg{i}': ccg[i-1] for i in range(1, self.last_ccg_len + 1)})
        hud(prompt=prompt)

        w, h = self.get_dimensions(w, h, image)
        image = self.get_images(image)
        # latents = self.step_latent_noise(w, h, seed, chg, seed_grain)
        latents = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, self.get_generator(seed))
        text_embed = self.compel.build_conditioning_tensor(prompt)
        text_embed_neg = self.compel.build_conditioning_tensor(promptneg or '')

        images = self.pcontrolnet(
                prompt_embeds=text_embed,
                negative_prompt_embeds=text_embed_neg,
                image=image,
                width=w,
                height=h,
                guidance_scale=cfg,
                controlnet_conditioning_scale=ccg,
                output_type='np',
                num_inference_steps=int(steps),
                latents=latents).images

        image = images[0]
        image = (image * 255).round().astype("uint8")

        text_embed = None
        text_embed_neg = None
        gc.collect()

        return image

    @plugfun(plugfun_redirect('sd_diffusers', 'txt2img_cn_dev'))
    @trace_decorator
    @trace_decorator_noarg
    def img2img_cn(self,
                   prompt: str = None,
                   promptneg: str = None,
                   cfg: float = 7.0,
                   image: any = None,
                   guidance: any = None,
                   ccg: float = 1.0,
                   chg: float = 0.01,
                   steps: int = 30,
                   seed: int = 0,
                   w: int = 512,
                   h: int = 512,
                   seed_grain: float = 0,
                   **kwargs):
        from src import renderer
        session = renderer.session

        if isinstance(ccg, (tuple, list)):
            self.last_ccg_len = len(ccg)
        else:
            ccg = [ccg] * self.last_ccg_len

        hud(chg=chg, cfg=cfg, seed=seed)
        hud(**{f'ccg{i}': ccg[i-1] for i in range(1, self.last_ccg_len + 1)})
        hud(prompt=prompt)

        w, h = self.get_dimensions(w, h, image)
        image = self.get_images(image)
        guidance = self.get_images(guidance)
        latents = self.step_latent_noise(w, h, seed, 0, seed_grain)
        text_embed = self.compel.build_conditioning_tensor(prompt)
        text_embed_neg = self.compel.build_conditioning_tensor(promptneg or '')

        print("img2img_cn", type(image), image)
        print("img2img_cn", type(guidance), guidance)
        # image = image.unsqueeze(0)

        images = self.pcontrolnet_img(
                prompt_embeds=text_embed,
                negative_prompt_embeds=text_embed_neg,
                image=image,
                control_image=guidance,
                width=w,
                height=h,
                strength=chg,
                guidance_scale=cfg,
                controlnet_conditioning_scale=ccg,
                output_type='np',
                num_inference_steps=int(steps),
                latents=latents).images

        image = images[0]
        image = (image * 255).round().astype("uint8")

        text_embed = None
        text_embed_neg = None
        gc.collect()

        return image

    def step_latent_noise(self, w, h, seed, chg, grain):
        generator = self.get_generator(seed)

        if self.latents is None:
            self.latents = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, generator)

        target = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, generator)
        latents = self.latents.cpu().numpy()
        target = target.cpu().numpy()
        latents = maths.arr_slerp(latents, target, chg)
        self.latents = torch.from_numpy(latents).to(device).to(torch.float16)

        # Add some base grain
        latents = tricks.grain_mul(latents, grain)
        latents = torch.from_numpy(latents).to(device).to(torch.float16)

        return latents

    @plugfun(plugfun_img)
    @trace_decorator
    def vd_var(self,
               cfg: float = 7.5,
               image: str = None,
               ccg: float = 1.0,
               chg: float = 0.5,
               steps: int = 30,
               seed: int = 0,
               **kwargs):
        import torch
        self.init_versatile()
        if image is not None:
            image = load_pil(image)
        if image is None:
            image = self.session.image

        # tform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(
        #             (224, 224),
        #             interpolation=transforms.InterpolationMode.BICUBIC,
        #             antialias=False,
        #     ),
        #     transforms.Normalize(
        #             [0.48145466, 0.4578275, 0.40821073],
        #             [0.26862954, 0.26130258, 0.27577711]),
        # ])
        # inp = tform(image).to(device).unsqueeze(0)
        #
        # out = self.pvar(inp, guidance_scale=cfg, num_inference_steps=steps)
        # ret = out["images"][0]

        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        result = self.pvd.image_variation(image,
                                          width=image.width,
                                          height=image.height,
                                          negative_prompt=None,
                                          guidance_scale=cfg,
                                          num_inference_steps=int(steps),
                                          generator=generator)
        ret = result["images"][0]

        return load_cv2(ret)

    @plugfun(plugfun_img)
    @trace_decorator
    def img2img_if(self,
                   prompt: str = None,
                   promptneg: str = None,
                   cfg: float = 7.0,
                   image: str = None,
                   ccg: float = 1.0,
                   chg: float = 0.5,
                   steps: int = 30,
                   seed: int = 0,
                   w: int = 512,
                   h: int = 512,
                   **kwargs):
        self.init_if()
        import torch
        from src import renderer

        session = renderer.session

        if not w: w = session.w
        if not h: h = session.h
        if not w and image.width: w = image.width
        if not h and image.height: h = image.height

        if isinstance(image, list):
            image = [load_pil(i) for i in image]
        else:
            image = load_pil(image)

        hud(prompt=prompt)
        hud(chg=chg, cfg=cfg, seed=seed)

        # text_embed = self.compel.build_conditioning_tensor(prompt)
        # text_embed_neg = self.compel.build_conditioning_tensor(promptneg or '')
        prompt_embeds, negative_embeds = self.pif1.encode_prompt(prompt)
        generator = self.get_generator(seed)
        original_image = image

        ret1 = self.pif1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                # prompt=prompt,
                # negative_prompt=promptneg or '',
                image=image,
                strength=chg,
                guidance_scale=cfg,
                output_type='np',
                num_inference_steps=int(steps),
                generator=generator).images

        ret2 = self.pif2(
                image=ret1,
                original_image=original_image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                generator=generator,
                output_type="pt",
        ).images

        from diffusers.utils import pt_to_pil
        ret3 = pt_to_pil(ret2)[0]
        ret3 = self.pif3(prompt=prompt, image=ret3, generator=generator, noise_level=100).images
        # ret3 = (ret3 * 255).round().astype("uint8")

        if ret3 is None or len(ret3) == 0:
            from src.renderer import rv
            return rv.img

        return ret3[0]

    @trace_decorator_noarg
    def txt2img_cn_dev(self, *args, **kwargs):
        hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
        if 'image' in kwargs:
            img = tricks.grain(rv, strength=kwargs['chg'])
            img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
            return img
        else:
            # random color
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
            return img

    @trace_decorator_noarg
    def txt2img_dev(self, *args, **kwargs):
        hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
        return img

    @trace_decorator_noarg
    def img2img_dev(self, *args, **kwargs):
        hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
        img = tricks.grain(rv, strength=kwargs['chg'])
        img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
        return img
