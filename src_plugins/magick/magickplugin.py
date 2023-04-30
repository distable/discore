import os

import cv2
# import wand.image
from PIL import Image

from src.classes import paths
from src.classes.convert import pil2cv
from src.classes.Plugin import Plugin
from src.installer import run
from src.party.maths import clamp


class MagickPlugin(Plugin):
    def title(self):
        return "Magick Plugin"

    def describe(self):
        return "Run some color corrections with image magick"

    def init(self):
        pass

    def install(self):
        # TODO this functionality should be in installing.py (with proper user input and stuff)
        # check if pacman is installed
        if run("pacman") == 0:
            # Install imagemagick if not installed
            if run("pacman -Q imagemagick") != 0:
                run("pacman -S imagemagick")

        # try with apt-get
        elif run("apt-get") == 0:
            # Install imagemagick if not installed
            if run("apt-get -Q imagemagick") != 0:
                run("apt-get install imagemagick")

        # oh well
        else:
            print("No package manager found to install ImageMagick")
            return

        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def get_avg_hsv(self, pil) -> (float, float, float):
        img_hsv = cv2.cvtColor(pil2cv(pil), cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0].mean() / 255
        sat = img_hsv[:, :, 1].mean() / 255
        val = img_hsv[:, :, 2].mean() / 255
        return hue, sat, val

    def magick(self, pil, command="+sigmoidal-contrast 5x-3%"):
        src = (paths.root / "_magick.png").as_posix()
        dst = (paths.root / "_out.png").as_posix()

        pil.save(src)
        os.system(f"convert '{src}' {command} '{dst}'")
        ret = Image.open('_out.png').convert('RGB')
        os.remove(src)
        os.remove(dst)

        return ret

    def __call__(self, pil, *args, **kwargs):
        return self.magick(pil, *args, **kwargs)

    def add_hsv(self, hue, img, sat, val):
        img_hsv = cv2.cvtColor(pil2cv(img), cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 0] += int(hue % 255)
        img_hsv[:, :, 1] += int(clamp(sat, -255, 255))
        img_hsv[:, :, 2] += int(clamp(val, -255, 255))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img

    # def denoise(self, j: denoise_job):
    #     if j.session.image is None:
    #         return None
    #
    #     return self.magick(j.session.image, f'-enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance -enhance')



    # def distort(self, j: distort_job):
    #     if j.session.image is None:
    #         return None
    #
    #     with wand.image.Image.from_array(np.array(j.session.image)) as img:
    #         # Grid distortion
    #         return wnd_to_pil(img)





def cv2_hue(c, amount):
    amount * 255
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c


def cv2_brightness(input_img, brightness=0):
    """
        input_image:  color or grayscale image
        brightness:  -127 (all black) to +127 (all white)
            returns image of same type as input_image but with
            brightness adjusted
    """
    brightness *= 127

    img = input_img.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        cv2.convertScaleAbs(input_img, img, alpha_b, gamma_b)

    return img
