import cv2
import numpy as np
import torch
import torch.nn as nn

from src.classes import paths
from src.lib.loglib import trace_decorator
from src.party.models import flow_utils
from src.renderer import rv
from src.rendering import hud
# from src_plugins.flower.flow_utils import frames_norm, flow_renorm, occl_renorm

DEVICE = 'cuda'
MODEL_H, MODEL_W = 512, 768  # Fixed dimensions for FloweR model
MODEL_PATH = 'FloweR/FloweR_0.1.1.pth'  # Default model path

# Global variables
model = None
clip_frames = None
original_h, original_w = None, None

# w, h = -1, -1
# model: nn.Module = None
install_location = paths.root_models / "flowr" / "FloweR_0.1.pth"


# clip_frames = None


def _init_model(orig_h, orig_w):
    """
    Initialize or reinitialize the FloweR model and clip_frames buffer.

    Args:
    orig_h (int): Original image height
    orig_w (int): Original image width
    """
    if not install_location.exists():
        print("Missing flower model. Download from https://drive.google.com/uc?export=download&id=1WhzoVIw6Kdg4EjfK9LaTLqFm5dF-IJ7F")
        return

    global model, clip_frames, original_h, original_w
    model = FloweR(input_size=(MODEL_H, MODEL_W))
    model.load_state_dict(torch.load(install_location))
    model = model.to(DEVICE)
    model.eval()

    # model = torch.compile(model)

    init_clip_frames()
    original_h, original_w = orig_h, orig_w


def init_clip_frames():
    global clip_frames
    clip_frames = np.zeros((4, MODEL_H, MODEL_W, 3), dtype=np.uint8)


def _push_image(img):
    """
    Resize and push a new image to the clip_frames buffer.

    Args:
    img (np.array): New image to be added to the buffer
    """
    global clip_frames
    resized_img = cv2.resize(img, (MODEL_W, MODEL_H))
    clip_frames = np.roll(clip_frames, -1, axis=0)
    clip_frames[-1] = resized_img


@trace_decorator
def get_flow(img):
    """Predict flow using the FloweR model."""
    global model, clip_frames, original_h, original_w

    if model is None or original_h != img.shape[0] or original_w != img.shape[1]:
        _init_model(img.shape[0], img.shape[1])
    # _init_model(img.shape[0], img.shape[1])

    init_clip_frames()

    # _push_image(rv.session.res_frame_cv2(rv.f - 1))
    _push_image(img)

    clip_frames_torch = flow_utils.frames_norm(torch.from_numpy(clip_frames).to(DEVICE, dtype=torch.float32))

    with torch.no_grad():
        pred_data = model(clip_frames_torch.unsqueeze(0))[0]

    pred_flow = flow_utils.flow_renorm(pred_data[..., :2]).cpu().numpy()
    pred_occl = flow_utils.occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)

    pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05)
    pred_flow = cv2.GaussianBlur(pred_flow, (71, 71), 1, cv2.BORDER_REFLECT_101)

    pred_occl = cv2.GaussianBlur(pred_occl, (51, 51), 2, cv2.BORDER_REFLECT_101)
    pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
    pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)

    # Scale flow back to original image size
    scale_h, scale_w = original_h / MODEL_H, original_w / MODEL_W
    pred_flow_scaled = cv2.resize(pred_flow, (original_w, original_h))
    pred_flow_scaled[..., 0] *= scale_w
    pred_flow_scaled[..., 1] *= scale_h

    pred_occl_scaled = cv2.resize(pred_occl, (original_w, original_h))

    return pred_flow_scaled, pred_occl_scaled


@trace_decorator
def apply_strength_and_rotation(flow, strength=1.0, rotation=0.0):
    """Apply strength scaling and rotation to the flow field."""
    # Convert rotation to radians
    rotation_rad = np.deg2rad(rotation)

    # Create rotation matrix
    rot_mat = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                        [np.sin(rotation_rad), np.cos(rotation_rad)]])

    # Reshape flow for matrix multiplication
    flow_reshaped = flow.reshape(-1, 2)

    # Apply rotation
    flow_rotated = np.dot(flow_reshaped, rot_mat.T)

    # Apply strength
    flow_scaled = flow_rotated * strength

    # Reshape back to original shape
    return flow_scaled.reshape(flow.shape)


@trace_decorator
def flow(img, strength=1.0, rotation=0.0):
    """
    Predict flow, apply strength and rotation, and use it to deform the input image.

    Args:
    img (np.array): Input image to be deformed
    strength (float): Factor to scale the motion (default: 1.0)
    rotation (float): Angle in degrees to rotate the motion field (default: 0.0)

    Returns:
    tuple: Deformed image, predicted flow, and occlusion mask
    """
    global original_h, original_w
    pred_flow, pred_occl = get_flow(img)

    # Apply strength and rotation to the flow
    modified_flow = apply_strength_and_rotation(pred_flow, strength, rotation)

    # Create the flow map in the correct format for cv2.remap
    flow_map = np.zeros((original_h, original_w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(original_w) + modified_flow[:, :, 0]
    flow_map[:, :, 1] = np.arange(original_h)[:, np.newaxis] + modified_flow[:, :, 1]

    # Ensure the image is in the correct format (8-bit unsigned integer)
    img_8bit = img.astype(np.uint8)

    warped_frame = cv2.remap(img_8bit, flow_map, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    hud.hud(flower=strength, flower_rot=rotation)

    return warped_frame  # , modified_flow, pred_occl


# ------------------------------------------------------------
# The model
# ------------------------------------------------------------


# Define the model
class FloweR(nn.Module):
    def __init__(self, input_size=(384, 384), window_size=4):
        super(FloweR, self).__init__()

        self.input_size = input_size
        self.window_size = window_size

        # INPUT: 384 x 384 x 10 * 3

        ### DOWNSCALE ###
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3 * self.window_size, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 384 x 384 x 128

        self.conv_block_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 192 x 192 x 128

        self.conv_block_3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 96 x 96 x 128

        self.conv_block_4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 48 x 48 x 128

        self.conv_block_5 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 24 x 24 x 128

        self.conv_block_6 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 12 x 12 x 128

        self.conv_block_7 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 6 x 6 x 128

        self.conv_block_8 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 3 x 3 x 128

        ### UPSCALE ###
        self.conv_block_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 6 x 6 x 128

        self.conv_block_10 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 12 x 12 x 128

        self.conv_block_11 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 24 x 24 x 128

        self.conv_block_12 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 48 x 48 x 128

        self.conv_block_13 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 96 x 96 x 128

        self.conv_block_14 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 192 x 192 x 128

        self.conv_block_15 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # 384 x 384 x 128

        self.conv_block_16 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        if x.size(1) != self.window_size:
            raise Exception(f'Shape of the input is not compatable. There should be exactly {self.window_size} frames in an input video.')

        # batch, frames, height, width, colors
        in_x = x.permute((0, 1, 4, 2, 3))
        # batch, frames, colors, height, width

        in_x = in_x.reshape(-1, self.window_size * 3, self.input_size[0], self.input_size[1])

        ### DOWNSCALE ###
        block_1_out = self.conv_block_1(in_x)  # 384 x 384 x 128
        block_2_out = self.conv_block_2(block_1_out)  # 192 x 192 x 128
        block_3_out = self.conv_block_3(block_2_out)  # 96 x 96 x 128
        block_4_out = self.conv_block_4(block_3_out)  # 48 x 48 x 128
        block_5_out = self.conv_block_5(block_4_out)  # 24 x 24 x 128
        block_6_out = self.conv_block_6(block_5_out)  # 12 x 12 x 128
        block_7_out = self.conv_block_7(block_6_out)  # 6 x 6 x 128
        block_8_out = self.conv_block_8(block_7_out)  # 3 x 3 x 128

        ### UPSCALE ###
        block_9_out = block_7_out + self.conv_block_9(block_8_out)  # 6 x 6 x 128
        block_10_out = block_6_out + self.conv_block_10(block_9_out)  # 12 x 12 x 128
        block_11_out = block_5_out + self.conv_block_11(block_10_out)  # 24 x 24 x 128
        block_12_out = block_4_out + self.conv_block_12(block_11_out)  # 48 x 48 x 128
        block_13_out = block_3_out + self.conv_block_13(block_12_out)  # 96 x 96 x 128
        block_14_out = block_2_out + self.conv_block_14(block_13_out)  # 192 x 192 x 128
        block_15_out = block_1_out + self.conv_block_15(block_14_out)  # 384 x 384 x 128

        block_16_out = self.conv_block_16(block_15_out)  # 384 x 384 x (2 + 1)
        out = block_16_out.reshape(-1, 3, self.input_size[0], self.input_size[1])

        # batch, colors, height, width
        out = out.permute((0, 2, 3, 1))
        # batch, height, width, colors
        return out

# from functools import lru_cache
#
# import cv2
# import numpy as np
# from src.party import maths
# import math
#
# from src.classes import paths
# from src.classes.convert import load_cv2
# from src.lib.devices import device
# from src.lib.loglib import trace_decorator
# from src.party import tricks, flow_viz
# from src.plugins import plugfun, plugfun_img
# from src.rendering import hud
# from src_plugins.flower.flow_utils import flow_renorm, frames_norm, occl_renorm
#
# # from FloweR.model import FloweR
# # from FloweR.utils import flow_viz
# # from FloweR import flow_utils
#
# import torch
# import torch.nn as nn
#
# w, h = -1, -1
# model: nn.Module = None
# install_location = paths.root_models / "flowr" / "FloweR_0.1.pth"
# clip_frames = None
#
#
# def init(height, width):
#     # installer.gdown(install_location, "https://drive.google.com/uc?export=download&id=1WhzoVIw6Kdg4EjfK9LaTLqFm5dF-IJ7F")
#     if not install_location.exists():
#         print("Missing flower model. Download from https://drive.google.com/uc?export=download&id=1WhzoVIw6Kdg4EjfK9LaTLqFm5dF-IJ7F")
#         return
#
#     global w, h
#     global model
#     global clip_frames
#
#     if w != width or h != height:
#         w = width
#         h = height
#         clip_frames = np.zeros((4, height, width, 3), dtype=np.uint8)
#         model = FloweR(input_size=(height, width))
#         model.load_state_dict(torch.load(install_location))
#         # Move the model to the device
#         model = model.to(device)
#         print("FlowerR model loaded.")
#
#
# def push(img):
#     global clip_frames
#
#     img = load_cv2(img)
#
#     init(img.shape[0], img.shape[1])
#
#     clip_frames = np.roll(clip_frames, -1, axis=0)
#     clip_frames[-1] = img
#
#
# @lru_cache(maxsize=32)
# def get_flow_cached(image_hash, strength):
#     # Implement caching logic for get_flow
#     # This is a placeholder and should be replaced with actual implementation
#     return get_flow(image_hash, strength)
#
#
# def normalize_flow(flow, shape):
#     h, w = shape
#     flow_normalized = flow.clone()
#     flow_normalized[:, :, 0] = flow[:, :, 0] / flow[:, :, 0].max() * (w - 1)
#     flow_normalized[:, :, 1] = flow[:, :, 1] / flow[:, :, 1].max() * (h - 1)
#     return flow_normalized
#
#
# @torch.jit.script
# def warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
#     n, c, h, w = image.shape
#     grid = torch.zeros(n, h, w, 2, device=image.device)
#     grid[:, :, :, 0] = torch.arange(w, device=image.device).view(1, 1, w).repeat(n, h, 1)
#     grid[:, :, :, 1] = torch.arange(h, device=image.device).view(1, h, 1).repeat(n, 1, w)
#     grid += flow.permute(0, 2, 3, 1)
#     grid[:, :, :, 0] = 2 * grid[:, :, :, 0] / (w - 1) - 1
#     grid[:, :, :, 1] = 2 * grid[:, :, :, 1] / (h - 1) - 1
#     return torch.nn.functional.grid_sample(image, grid, mode='bicubic', padding_mode='reflection', align_corners=False)
#
#
# # @plugfun(plugfun_img)
# @trace_decorator
# def flow(image, strength=1, flow=None, rotation_angle=0):
#     if image is None:
#         return None
#
#     h, w = image.shape[:2]
#
#     if flow is None:
#         flow = get_flow(image, strength)
#
#     hud.hud(flower=strength, rotation=rotation_angle)
#
#     # Use numpy operations for faster coordinate normalization
#     new_xs = flow[:, :, 0]
#     new_ys = flow[:, :, 1]
#     new_xs = np.clip(new_xs / np.max(np.abs(new_xs)) * (w - 1), 0, w - 1)
#     new_ys = np.clip(new_ys / np.max(np.abs(new_ys)) * (h - 1), 0, h - 1)
#
#     # Apply rotation to the flow
#     rotated_flow = rotate_flow(flow, rotation_angle)
#
#     # Convert to float32 for faster processing
#     new_xs = rotated_flow[:, :, 0].astype(np.float32)
#     new_ys = rotated_flow[:, :, 1].astype(np.float32)
#
#     warped_frame = cv2.remap(image,
#                              new_xs,
#                              new_ys,
#                              cv2.INTER_CUBIC,
#                              borderMode=cv2.BORDER_REFLECT_101)
#
#     hud.snap('flower_before', image)
#     hud.snap('flower_warped', warped_frame)
#     return warped_frame
#
# def rotate_flow(flow, angle_degrees):
#     """
#     Rotate the flow field by the given angle in degrees.
#
#     :param flow: The input flow field (h, w, 2)
#     :param angle_degrees: Rotation angle in degrees
#     :return: Rotated flow field
#     """
#     angle_radians = math.radians(angle_degrees)
#     cos_theta = math.cos(angle_radians)
#     sin_theta = math.sin(angle_radians)
#
#     # Create rotation matrix
#     rotation_matrix = np.array([
#         [cos_theta, -sin_theta],
#         [sin_theta, cos_theta]
#     ])
#
#     # Reshape flow for matrix multiplication
#     original_shape = flow.shape
#     flow_reshaped = flow.reshape(-1, 2)
#
#     # Apply rotation
#     rotated_flow = np.dot(flow_reshaped, rotation_matrix.T)
#
#     # Reshape back to original shape
#     return rotated_flow.reshape(original_shape)
#
#
# def get_flow(image, strength=1.0, as_img=False):
#     img_w = image.shape[1]
#     img_h = image.shape[0]
#
#     # ww = img_w // 128 * 128
#     # hh = img_h // 128 * 128
#
#     ww = 512
#     hh = 512
#
#
#     init(hh, ww)
#     im = cv2.resize(image, (ww, hh), interpolation=cv2.INTER_CUBIC)
#     push(im)
#
#     # Run inference on frames
#     clip_frames_torch = frames_norm(torch.from_numpy(clip_frames).to(device, dtype=torch.float32))
#     with torch.no_grad():
#         pred_data = model(clip_frames_torch.unsqueeze(0))[0]
#
#     pred_flow = flow_renorm(pred_data[..., :2]).cpu().numpy()
#     pred_occl = occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)
#     pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05)
#
#     # Post process the flow and occlusion
#     pred_flow = cv2.GaussianBlur(pred_flow, (31, 31), 1, cv2.BORDER_REFLECT_101)
#     pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
#     pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
#     pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)
#     flow_map = pred_flow.copy()
#     flow_map = tricks.cancel_global_motion(flow_map)
#
#     # Apply strength
#     flow_map *= strength
#
#     # Scale the flow map to the image size
#     flow_map[:, :, 0] += np.arange(w)
#     flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
#
#     flow_map = cv2.resize(flow_map,
#                           (image.shape[1], image.shape[0]),
#                           flow_map,
#                           interpolation=cv2.INTER_LINEAR)
#
#     if as_img:
#         # Flow image
#         # frames_img = cv2.hconcat(list(self.clip_frames))
#         # data_img = cv2.hconcat([flow_img, pred_occl, warped_frame])
#         flow_img = flow_viz.flow_to_image(pred_flow)
#         return flow_img
#
#     hud.snap('flower_flow', flow_map)
#     return flow_map
#
# # --------
#
#
# def get_flow(image):
#     """
#     Predict flow using the FloweR model.
#
#     Args:
#     model (FloweR): The FloweR model instance
#     clip_frames (np.array): Input frames of shape (4, h, w, 3)
#
#     Returns:
#     tuple: Predicted flow and occlusion mask
#     """
#     clip_frames_torch = frames_norm(torch.from_numpy(clip_frames).to(DEVICE, dtype=torch.float32))
#
#     with torch.no_grad():
#         pred_data = model(clip_frames_torch.unsqueeze(0))[0]
#
#     pred_flow = flow_renorm(pred_data[..., :2]).cpu().numpy()
#     pred_occl = occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)
#
#     pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05)
#     pred_flow = cv2.GaussianBlur(pred_flow, (31, 31), 1, cv2.BORDER_REFLECT_101)
#
#     pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
#     pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
#     pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)
#
#     return pred_flow, pred_occl
#
#
# def flow(model, img, clip_frames):
#     """
#     Predict flow and apply it to deform the input image.
#
#     Args:
#     model (FloweR): The FloweR model instance
#     img (np.array): Input image to be deformed
#     clip_frames (np.array): Input frames of shape (4, h, w, 3)
#
#     Returns:
#     tuple: Deformed image, predicted flow, and occlusion mask
#     """
#     pred_flow, pred_occl = get_flow(model, clip_frames)
#
#     flow_map = pred_flow.copy()
#     flow_map[:, :, 0] += np.arange(w)
#     flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
#
#     warped_frame = cv2.remap(img, flow_map, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
#
#     return warped_frame, pred_flow, pred_occl
#
#
#
#
# # ------------------------------------------------------------
# # The model
# # ------------------------------------------------------------
#
#
# # Define the model
# class FloweR(nn.Module):
#     def __init__(self, input_size=(384, 384), window_size=4):
#         super(FloweR, self).__init__()
#
#         self.input_size = input_size
#         self.window_size = window_size
#
#         # INPUT: 384 x 384 x 10 * 3
#
#         ### DOWNSCALE ###
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(3 * self.window_size, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 384 x 384 x 128
#
#         self.conv_block_2 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 192 x 192 x 128
#
#         self.conv_block_3 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 96 x 96 x 128
#
#         self.conv_block_4 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 48 x 48 x 128
#
#         self.conv_block_5 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 24 x 24 x 128
#
#         self.conv_block_6 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 12 x 12 x 128
#
#         self.conv_block_7 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 6 x 6 x 128
#
#         self.conv_block_8 = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 3 x 3 x 128
#
#         ### UPSCALE ###
#         self.conv_block_9 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 6 x 6 x 128
#
#         self.conv_block_10 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 12 x 12 x 128
#
#         self.conv_block_11 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 24 x 24 x 128
#
#         self.conv_block_12 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 48 x 48 x 128
#
#         self.conv_block_13 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 96 x 96 x 128
#
#         self.conv_block_14 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 192 x 192 x 128
#
#         self.conv_block_15 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
#             nn.ReLU(),
#         )  # 384 x 384 x 128
#
#         self.conv_block_16 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding='same')
#
#     def forward(self, x):
#         if x.size(1) != self.window_size:
#             raise Exception(f'Shape of the input is not compatable. There should be exactly {self.window_size} frames in an input video.')
#
#         # batch, frames, height, width, colors
#         in_x = x.permute((0, 1, 4, 2, 3))
#         # batch, frames, colors, height, width
#
#         in_x = in_x.reshape(-1, self.window_size * 3, self.input_size[0], self.input_size[1])
#
#         ### DOWNSCALE ###
#         block_1_out = self.conv_block_1(in_x)  # 384 x 384 x 128
#         block_2_out = self.conv_block_2(block_1_out)  # 192 x 192 x 128
#         block_3_out = self.conv_block_3(block_2_out)  # 96 x 96 x 128
#         block_4_out = self.conv_block_4(block_3_out)  # 48 x 48 x 128
#         block_5_out = self.conv_block_5(block_4_out)  # 24 x 24 x 128
#         block_6_out = self.conv_block_6(block_5_out)  # 12 x 12 x 128
#         block_7_out = self.conv_block_7(block_6_out)  # 6 x 6 x 128
#         block_8_out = self.conv_block_8(block_7_out)  # 3 x 3 x 128
#
#         ### UPSCALE ###
#         block_9_out = block_7_out + self.conv_block_9(block_8_out)  # 6 x 6 x 128
#         block_10_out = block_6_out + self.conv_block_10(block_9_out)  # 12 x 12 x 128
#         block_11_out = block_5_out + self.conv_block_11(block_10_out)  # 24 x 24 x 128
#         block_12_out = block_4_out + self.conv_block_12(block_11_out)  # 48 x 48 x 128
#         block_13_out = block_3_out + self.conv_block_13(block_12_out)  # 96 x 96 x 128
#         block_14_out = block_2_out + self.conv_block_14(block_13_out)  # 192 x 192 x 128
#         block_15_out = block_1_out + self.conv_block_15(block_14_out)  # 384 x 384 x 128
#
#         block_16_out = self.conv_block_16(block_15_out)  # 384 x 384 x (2 + 1)
#         out = block_16_out.reshape(-1, 3, self.input_size[0], self.input_size[1])
#
#         # batch, colors, height, width
#         out = out.permute((0, 2, 3, 1))
#         # batch, height, width, colors
#         return out# import torch
