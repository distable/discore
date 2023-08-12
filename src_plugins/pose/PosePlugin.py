from mmpose.apis import MMPoseInferencer

from src.classes.Plugin import Plugin

# from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
#                          process_mmdet_results, vis_pose_result)
# from mmpose.datasets import DatasetInfo
# from mmdet.apis import inference_detector, init_detector
#
# det_model = init_detector(
#         "./external/faster_rcnn_r50_fpn_coco.py",
#         "./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
#         device="cpu")
# pose_model = init_pose_model(
#         "./external/hrnet_w48_coco_256x192.py",
#         "./hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
#         device="cpu")
#
# dataset = pose_model.cfg.data['test']['type']
# dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
# dataset_info = DatasetInfo(dataset_info)

class PosePlugin(Plugin):
    def title(self):
        return "pose"

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
        # self.inferencer = MMPoseInferencer('human')

    def unload(self):
        pass

    # def img_to_pose(self, image):
    #     return self.inferencer(image).get('predictions')
    #
    # def img_to_img(self, image):
    #     result = self.inferencer(image).get("visualization")

    # def infer(self,image):
    #     mmdet_results = inference_detector(det_model, image)
    #     person_results = process_mmdet_results(mmdet_results, 1)
    #
    #     pose_results, returned_outputs = inference_top_down_pose_model(
    #             pose_model,
    #             image,
    #             person_results,
    #             bbox_thr=0.3,
    #             format='xyxy',
    #             dataset=dataset,
    #             dataset_info=dataset_info,
    #             return_heatmap=False,
    #             outputs=None)
    #
    #     return pose_results, returned_outputs
    #
    # def draw(self, image, results):
    #     return vis_pose_result(
    #             pose_model,
    #             image,
    #             results,
    #             dataset=dataset,
    #             dataset_info=dataset_info,
    #             kpt_score_thr=0.3,
    #             radius=4,
    #             thickness=3,
    #             show=False,
    #             out_file=None)
