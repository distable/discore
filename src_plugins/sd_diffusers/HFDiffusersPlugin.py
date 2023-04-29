import numpy as np
import tomesd
import torch
from compel import Compel

from classes.convert import load_cv2, load_pil
from lib.devices import device
from plugins import plugfun, plugfun_img
from src_core.rendering.hud import hud
from src_core.classes.Plugin import Plugin

from diffusers import AutoencoderKL, ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, UniPCMultistepScheduler

TOMESD_RATIO = 0.75

# This is the function if you want to use it
def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

class HFDiffusersPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ptxt = None
        self.pimg = None
        self.pvar = None
        self.pvd = None
        self.pcontrolnet = None
        self.pcontrolnet_inpaint = None
        self.controlnets = {}
        self.latents = None

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

    def sd(self):
        self.init_sd()
        return self

    def cn(self, *models):
        self.init_controlnet(models)
        return self

    @plugfun()
    def init_controlnet(self, models):
        def load_model(name):
            if name == "canny": return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", file='', torch_dtype=torch.float16)
            elif name == "hed":  return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", file='', torch_dtype=torch.float16)
            elif name == "depth":  return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", file='', torch_dtype=torch.float16)
            elif name == "temporal":  return ControlNetModel.from_pretrained("CiaraRowles/TemporalNet", file='', torch_dtype=torch.float16)


        print("Diffusers: Loading ControlNet...")
        model_list = []
        for model in models:
            if model not in self.controlnets:
                self.controlnets[model] = load_model(model)
            model_list.append(self.controlnets[model])

        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.pcontrolnet = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                requires_safety_checker=False,
                controlnet=model_list,
                torch_dtype=torch.float16).to("cuda")
        self.pcontrolnet.scheduler = UniPCMultistepScheduler.from_config(self.pcontrolnet.scheduler.config)
        self.compel = Compel(tokenizer=self.pcontrolnet.tokenizer, text_encoder=self.pcontrolnet.text_encoder)
        if self.pimg is not None:
            self.pimg.scheduler = self.pcontrolnet.scheduler

        tomesd.apply_patch(self.pcontrolnet.unet, ratio=TOMESD_RATIO, sx=2, sy=2, max_downsample=1)

        # if self.pcontrolnet is None:
        # else:
        #     print("Swapping controlnets...")
        #     self.pcontrolnet.controlnet = StableDiffusionControlNetPipeline.from_pretrained(
        #             "runwayml/stable-diffusion-v1-5",
        #             safety_checker=None,
        #             requires_safety_checker=False,
        #             controlnet=model_list,
        #             unet = self.pcontrolnet.unet,
        #             text_encoder = self.pcontrolnet.text_encoder,
        #             vae = self.pcontrolnet.vae,
        #             tokenizer = self.pcontrolnet.tokenizer,
        #             scheduler = self.pcontrolnet.scheduler,
        #             torch_dtype=torch.float16).to("cuda")

    @plugfun()
    def init_sd(self):
        if self.ptxt is not None: return
        print("Diffusers: Loading SD...")

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.ptxt = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            vae=vae,
                                                            safety_checker=None,
                                                            requires_safety_checker=False,
                                                            torch_dtype=torch.float32).to("cuda")
        self.pimg = StableDiffusionImg2ImgPipeline(
                vae=vae,
                text_encoder=self.ptxt.text_encoder,
                tokenizer=self.ptxt.tokenizer,
                unet=self.ptxt.unet,
                scheduler=self.ptxt.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False).to("cuda")
        self.compel = Compel(tokenizer=self.ptxt.tokenizer, text_encoder=self.ptxt.text_encoder)
        tomesd.apply_patch(self.ptxt.unet, ratio=TOMESD_RATIO)
        if self.pcontrolnet is not None:
            self.pimg.scheduler = self.pcontrolnet.scheduler


        # self.ptxt.enable_model_cpu_offload()
        # self.ptxt.enable_attention_slicing(1)
        # pipe.enable_vae_tiling()
        # pipe.enable_sequential_cpu_offload()
        # pipe.unet.to(memory_format=torch.channels_last)  # in-place operation


    @plugfun(plugfun_img)
    def txt2img_cn(self,
                   prompt: str = None,
                   negprompt: str = None,
                   cfg: float = 7.0,
                   image: str = None,
                   ccg: float = 1.0,
                   chg: float = 0.01,
                   steps: int = 30,
                   seed: int = 0,
                   w: int = 512,
                   h: int = 512,
                   seed_grain: float = 0,
                   **kwargs):
        from src_core.party import tricks
        import renderer
        session = renderer.session

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)
        # hud(image=image)

        if not w: w = session.w
        if not h: h = session.h
        if not w and image.width: w = image.width
        if not h and image.height: h = image.height

        if isinstance(image, (list, tuple)):
            image = [load_pil(i) for i in image]
        else:
            image = load_pil(image)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if self.latents is None:
            self.latents = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, generator)
        target = self.pcontrolnet.prepare_latents(1, self.pcontrolnet.unet.config.in_channels, h, w, torch.float16, device, generator)
        latents = self.latents.cpu().numpy()
        target = target.cpu().numpy()
        latents = slerp(latents, target, chg)

        self.latents = torch.from_numpy(latents).to(device).to(torch.float16)

        # Add some base grain
        latents = tricks.grain_mul(latents, seed_grain)
        latents = torch.from_numpy(latents).to(device).to(torch.float16)


        # text_embed = self.compel.build_conditioning_tensor(prompt)
        # text_embed_neg = self.compel.build_conditioning_tensor(negprompt or '')

        images = self.pcontrolnet(
                # prompt_embeds=text_embed,
                # negative_prompt_embeds=text_embed_neg,
                prompt=prompt,
                negative_prompt=negprompt or '',
                image=image or session.img,
                # rv.img = im.astype(np.uint8)
                width=w,
                height=h,
                guidance_scale=cfg,
                controlnet_conditioning_scale=ccg,
                output_type='np',
                num_inference_steps=int(steps),
                latents=latents).images

        image = images[0]
        image = (image * 255).round().astype("uint8")
        return image

    @plugfun(plugfun_img)
    def txt2img(self,
                prompt: str = None,
                negprompt: str = None,
                cfg: float = 7.0,
                image: str = None,
                ccg: float = 1.0,
                chg: float = 0.5,
                steps: int = 30,
                seed: int = 0,
                **kwargs):
        self.init_sd()
        import renderer
        session = renderer.session

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)
        # text_embed = self.compel.build_conditioning_tensor(prompt)
        # text_embed_neg = self.compel.build_conditioning_tensor(negprompt or '')

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        images = self.ptxt(
                # prompt_embeds=text_embed,
                # negative_prompt_embeds=text_embed_neg,
                prompt=prompt,
                negative_prompt=negprompt or '',
                image=image or session.img,
                guidance_scale=cfg,
                output_type='np',
                num_inference_steps=int(steps),
                generator=generator).images

        if images is None or len(images) == 0:
            from renderer import rv
            return rv.img

        image = images[0]
        image = (image * 255).round().astype("uint8")
        return image

    @plugfun(plugfun_img)
    def img2img(self,
                prompt: str = None,
                negprompt: str = None,
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
        import renderer
        session = renderer.session

        if not w: w = session.w
        if not h: h = session.h
        if not w and image.width: w = image.width
        if not h and image.height: h = image.height

        if isinstance(image, list):
            image = [load_pil(i) for i in image]
        else:
            image = load_pil(image)

        hud(seed=seed, chg=chg, cfg=cfg, ccg=ccg)
        hud(prompt=prompt)

        # text_embed = self.compel.build_conditioning_tensor(prompt)
        # text_embed_neg = self.compel.build_conditioning_tensor(negprompt or '')
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        images = self.pimg(
                # prompt_embeds=text_embed,
                # negative_prompt_embeds=text_embed_neg,
                prompt=prompt,
                negative_prompt=negprompt or '',
                image=image,
                strength=chg,
                guidance_scale=cfg,
                output_type='np',
                num_inference_steps=int(steps),
                generator=generator).images

        if images is None or len(images) == 0:
            from renderer import rv
            return rv.img

        image = images[0]
        image = (image * 255).round().astype("uint8")

        return image

    @plugfun(plugfun_img)
    def vd_var(self,
               cfg: float = 7.5,
               image: str = None,
               ccg: float = 1.0,
               chg: float = 0.5,
               steps: int = 30,
               seed: int = 0,
               **kwargs):
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
