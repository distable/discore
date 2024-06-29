import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plug_repos.unimatch.unimatch.dataloader.depth import augmentation
from plug_repos.unimatch.unimatch.unimatch.unimatch import UniMatch
from plug_repos.unimatch.unimatch.utils.utils import InputPadder

from src.classes.convert import load_cv2, load_torch
from src.classes.Plugin import Plugin
from src.lib.devices import device
from src.lib.loglib import trace_decorator

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEPTH_CKPT_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-demon-7c23f230.pth'
# FLOW_CKPT_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
FLOW_CKPT_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth'
# FLOW_CKPT_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth'
FLOW_CKPT_UPSAMPLE = 4
FLOW_CKPT_SCALE = 2
FLOW_CKPT_ATTN_SPLITS_LIST = [4, 16]
FLOW_CKPT_CORR_RADIUS_LIST = [-1, 4]
FLOW_CKPT_PROP_RADIUS_LIST = [-1, 1]
FLOW_CKPT_REG_REFINE = False
FLOW_CKPT_REG_REFINE_NUM = 1
FLOW_CKPT_PADDING_FACTOR = 8
FLOW_CKPT_ATTN_TYPE = 'swin'

class UnimatchPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_model = None
        self.flow_model = None

    def title(self):
        return "unimatch"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    # def load(self):
    #     self.model = self.load_unimatch_model()

    def unload(self):
        pass


    def load_unimatch_model(self, ckpt_path, strict_resume=False, feature_channels=128, num_scales=1, upsample_factor=4,
                            num_head=1, ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=False, task='depth'):
        model = UniMatch(feature_channels=feature_channels,
                         num_scales=num_scales,
                         upsample_factor=upsample_factor,
                         num_head=num_head,
                         ffn_dim_expansion=ffn_dim_expansion,
                         num_transformer_layers=num_transformer_layers,
                         reg_refine=reg_refine,
                         task=task)

        loc = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not 'http' in ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=loc)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(ckpt_path, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=strict_resume)

        if torch.cuda.is_available():
            model.cuda()

        torch.compile(model, mode='reduce-overhead', fullgraph=True)

        return model

    @trace_decorator
    def get_depth(self, image, padding_factor=16, inference_size=None, attn_type='swin',
                  attn_splits_list=[2], prop_radius_list=[-1], num_depth_candidates=64, num_reg_refine=1,
                  min_depth=0.5, max_depth=10, depth_from_argmax=False, pred_bidir_depth=False, output_path='output'):
        # Apply data augmentation
        transform = augmentation.Compose([
            augmentation.ToTensor(),
            augmentation.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        img_ref = transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            img_ref = img_ref.cuda()

        # Estimate depth
        results_dict = inference_depth(self.model,
                                       img_ref=img_ref,
                                       padding_factor=padding_factor,
                                       inference_size=inference_size,
                                       attn_type=attn_type,
                                       attn_splits_list=attn_splits_list,
                                       prop_radius_list=prop_radius_list,
                                       num_depth_candidates=num_depth_candidates,
                                       num_reg_refine=num_reg_refine,
                                       min_depth=min_depth,
                                       max_depth=max_depth,
                                       depth_from_argmax=depth_from_argmax,
                                       pred_bidir_depth=pred_bidir_depth,
                                       output_path=output_path)

        print(results_dict)


    @trace_decorator
    def get_flow(self,
                 image1,
                 image2,
                 inference_size=None):
        image1 = load_torch(image1).cuda()
        image2 = load_torch(image2).cuda()

        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)
        else:
            padder = InputPadder(image1.shape, padding_factor=FLOW_CKPT_PADDING_FACTOR)
            image1, image2 = padder.pad(image1, image2)

        if self.flow_model is None:
            self.flow_model = self.load_unimatch_model(FLOW_CKPT_URL,
                                                       num_scales=FLOW_CKPT_SCALE,
                                                       upsample_factor=FLOW_CKPT_UPSAMPLE,
                                                       reg_refine=FLOW_CKPT_REG_REFINE,
                                                       task='flow')

        print("Unimatch.get_flow: inference...")

        # image1.half()
        # image2.half()
        results_dict = self.flow_model(image1, image2,
                                       attn_type=FLOW_CKPT_ATTN_TYPE,
                                       attn_splits_list=FLOW_CKPT_ATTN_SPLITS_LIST,
                                       corr_radius_list=FLOW_CKPT_CORR_RADIUS_LIST,
                                       prop_radius_list=FLOW_CKPT_PROP_RADIUS_LIST,
                                       num_reg_refine=FLOW_CKPT_REG_REFINE_NUM,
                                       padding_factor=FLOW_CKPT_PADDING_FACTOR,
                                       task='flow')

        return results_dict['flow_preds'][-1].detach().cpu().numpy().squeeze().transpose(1, 2, 0)

@torch.no_grad()
def inference_depth(model,
                    inference_dir=None,
                    output_path='output',
                    padding_factor=16,
                    inference_size=None,
                    attn_type='swin',
                    attn_splits_list=None,
                    prop_radius_list=None,
                    num_reg_refine=1,
                    num_depth_candidates=64,
                    min_depth=0.5,
                    max_depth=10,
                    depth_from_argmax=False,
                    pred_bidir_depth=False,
                    ):
    model.eval()

    val_transform_list = [augmentation.ToTensor(),
                          augmentation.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = augmentation.Compose(val_transform_list)

    valid_samples = 0

    fixed_inference_size = inference_size

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # assume scannet dataset file structure
    imgs = sorted(glob(os.path.join(inference_dir, 'color', '*.jpg')) +
                  glob(os.path.join(inference_dir, 'color', '*.png')))
    poses = sorted(glob(os.path.join(inference_dir, 'pose', '*.txt')))

    intrinsics_file = glob(os.path.join(inference_dir, 'intrinsic', '*.txt'))[0]

    assert len(imgs) == len(poses)

    num_samples = len(imgs)

    for i in range(len(imgs) - 1):
        if i % 50 == 0:
            print('=> Predicting %d/%d' % (i, num_samples))

        img_ref = np.array(Image.open(imgs[i]).convert('RGB')).astype(np.float32)
        img_tgt = np.array(Image.open(imgs[i + 1]).convert('RGB')).astype(np.float32)

        intrinsics = np.loadtxt(intrinsics_file).astype(np.float32).reshape((4, 4))[:3, :3]  # [3, 3]

        pose_ref = np.loadtxt(poses[i], delimiter=' ').astype(np.float32).reshape((4, 4))
        pose_tgt = np.loadtxt(poses[i + 1], delimiter=' ').astype(np.float32).reshape((4, 4))
        # relative pose
        pose = np.linalg.inv(pose_tgt) @ pose_ref

        sample = {'img_ref'   : img_ref,
                  'img_tgt'   : img_tgt,
                  'intrinsics': intrinsics,
                  'pose'      : pose,
                  }
        sample = val_transform(sample)

        img_ref = sample['img_ref'].to(device).unsqueeze(0)  # [1, 3, H, W]
        img_tgt = sample['img_tgt'].to(device).unsqueeze(0)  # [1, 3, H, W]
        intrinsics = sample['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]
        pose = sample['pose'].to(device).unsqueeze(0)  # [1, 4, 4]

        nearest_size = [int(np.ceil(img_ref.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(img_ref.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = img_ref.shape[-2:]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            img_ref = F.interpolate(img_ref, size=inference_size, mode='bilinear',
                                    align_corners=True)
            img_tgt = F.interpolate(img_tgt, size=inference_size, mode='bilinear',
                                    align_corners=True)

        valid_samples += 1

        with torch.no_grad():
            pred_depth = model(img_ref, img_tgt,
                               attn_type=attn_type,
                               attn_splits_list=attn_splits_list,
                               prop_radius_list=prop_radius_list,
                               num_reg_refine=num_reg_refine,
                               intrinsics=intrinsics,
                               pose=pose,
                               min_depth=1. / max_depth,
                               max_depth=1. / min_depth,
                               num_depth_candidates=num_depth_candidates,
                               pred_bidir_depth=pred_bidir_depth,
                               depth_from_argmax=depth_from_argmax,
                               task='depth',
                               )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # resize back
            pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                       align_corners=True).squeeze(1)  # [1, H, W]

        pr_depth = pred_depth[0]

        filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '.png')
        viz_inv_depth = viz_depth_tensor(1. / pr_depth.cpu(),
                                         return_numpy=True)  # [H, W, 3] uint8
        Image.fromarray(viz_inv_depth).save(filename)

        if pred_bidir_depth:
            assert pred_depth.size(0) == 2

            pr_depth_bwd = pred_depth[1]

            filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '_bwd.png')
            viz_inv_depth = viz_depth_tensor(1. / pr_depth_bwd.cpu(),
                                             return_numpy=True)  # [H, W, 3] uint8
            Image.fromarray(viz_inv_depth).save(filename)

    print('Done!')

def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    import matplotlib as mpl
    import matplotlib.cm as cm

    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz
