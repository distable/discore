import sys

import cv2
import numpy as np
import torch
from PIL import Image

from src.classes import paths
from src.classes.convert import save_png
from src.party.tricks import session

sam_model = None


def segment(img, outpath):
	from fastsam import FastSAM, FastSAMPrompt

	global sam_model
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

	if sam_model is None:
		sam_model = FastSAM((paths.plug_res / 'fastsam.pt').as_posix())
		import matplotlib.pyplot as plt
		plt.switch_backend("Agg")

	everything_results = sam_model(img, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.05, iou=0.90)
	prompt_process = FastSAMPrompt(img, everything_results, device=DEVICE)
	ann = prompt_process.everything_prompt()

	# prompt_process.plot(annotations=ann, output_path=outpath, withContours=False)
	if len(ann) == 0:
		return None

	# Replace with a black img
	img = np.zeros_like(prompt_process.img)

	# Plot the masks
	result = plot_to_result(img, ann, withContours=False)
	result = result[:, :, ::-1]

	save_png(result, outpath, with_async=True)

	return ann


def precompute_sam(vid):
	dirname = f'.{vid.stem}.sam'
	for i, img, path in session.res_frameiter(vid, f"Precomputing segmentations for {vid} with FastSAM"):
		segment(img, session.res_frame(dirname, i).as_posix())

	return session.res(dirname)


# def precompute()


def plot_to_result(img,
                   annotations,
                   bboxes=None,
                   points=None,
                   point_label=None,
                   mask_random_color=True,
                   better_quality=True,
                   retina=False,
                   withContours=True) -> np.ndarray:
	import matplotlib.pyplot as plt

	if isinstance(annotations[0], dict):
		annotations = [annotation['segmentation'] for annotation in annotations]
	image = img
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	original_h = image.shape[0]
	original_w = image.shape[1]
	if sys.platform == "darwin":
		plt.switch_backend("TkAgg")
	plt.figure(figsize=(original_w / 100, original_h / 100))
	# Add subplot with no margin.
	plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
	plt.margins(0, 0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	plt.imshow(image)
	if better_quality:
		if isinstance(annotations[0], torch.Tensor):
			annotations = np.array(annotations.cpu())
		for i, mask in enumerate(annotations):
			mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
			annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

	if isinstance(annotations[0], np.ndarray):
		annotations = torch.from_numpy(annotations)
	fast_show_mask_gpu(
		annotations,
		plt.gca(),
		random_color=mask_random_color,
		bboxes=bboxes,
		points=points,
		pointlabel=point_label,
		retinamask=retina,
		target_height=original_h,
		target_width=original_w,
	)
	if isinstance(annotations, torch.Tensor):
		annotations = annotations.cpu().numpy()
	if withContours:
		contour_all = []
		temp = np.zeros((original_h, original_w, 1))
		for i, mask in enumerate(annotations):
			if type(mask) == dict:
				mask = mask['segmentation']
			annotation = mask.astype(np.uint8)
			if not retina:
				annotation = cv2.resize(
					annotation,
					(original_w, original_h),
					interpolation=cv2.INTER_NEAREST,
				)
			contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			for contour in contours:
				contour_all.append(contour)
		cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
		color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
		contour_mask = temp / 255 * color.reshape(1, 1, -1)
		plt.imshow(contour_mask)

	plt.axis('off')
	fig = plt.gcf()
	plt.draw()

	try:
		buf = fig.canvas.tostring_rgb()
	except AttributeError:
		fig.canvas.draw()
		buf = fig.canvas.tostring_rgb()
	cols, rows = fig.canvas.get_width_height()
	img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
	result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
	plt.close()
	return result


def fast_show_mask_gpu(
		annotation,
		ax,
		random_color=False,
		bboxes=None,
		points=None,
		pointlabel=None,
		retinamask=True,
		target_height=960,
		target_width=960,
):
	import matplotlib.pyplot as plt

	msak_sum = annotation.shape[0]
	height = annotation.shape[1]
	weight = annotation.shape[2]
	areas = torch.sum(annotation, dim=(1, 2))
	sorted_indices = torch.argsort(areas, descending=False)
	annotation = annotation[sorted_indices]
	# Find the index of the first non-zero value at each position.
	index = (annotation != 0).to(torch.long).argmax(dim=0)
	if random_color:
		palette = palette_ade20k[:msak_sum]
		palette_norm = np.array(palette) / 255
		# tensor_norm = np.expand_dims(palette_norm, axis=1)
		tensor_norm = np.expand_dims(np.expand_dims(palette_norm, axis=1), axis=1)
		tensor_norm = torch.from_numpy(tensor_norm).to(annotation.device)
		tensor_norm = tensor_norm.float()

		color = tensor_norm
		color_rand = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
	else:
		color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([
			30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
	transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 1.0
	visual = torch.cat([color, transparency], dim=-1)
	mask_image = torch.unsqueeze(annotation, -1) * visual
	# Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
	show = torch.zeros((height, weight, 4)).to(annotation.device)
	try:
		h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
	except:
		h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
	indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
	# Use vectorized indexing to update the values of 'show'.
	show[h_indices, w_indices, :] = mask_image[indices]
	show_cpu = show.cpu().numpy()
	if bboxes is not None:
		for bbox in bboxes:
			x1, y1, x2, y2 = bbox
			ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
	# draw point
	if points is not None:
		plt.scatter(
			[point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
			[point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
			s=20,
			c='y',
		)
		plt.scatter(
			[point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
			[point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
			s=20,
			c='m',
		)
	if not retinamask:
		show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
	ax.imshow(show_cpu)


def precompute_seg(vid):
	from transformers import AutoImageProcessor
	from transformers import UperNetForSemanticSegmentation
	image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
	image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

	for i, img, path in session.res_frameiter(vid, f"Precomputing segmentations for {vid}"):
		img = Image.fromarray(img)
		pixel_values = image_processor(img, return_tensors="pt").pixel_values

		with torch.no_grad():
			outputs = image_segmentor(pixel_values)

		seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]

		color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
		for label, color in enumerate(palette_ade20k):
			color_seg[seg == label, :] = color

		color_seg = color_seg.astype(np.uint8)
		image = Image.fromarray(color_seg)

		save_png(image, session.res_frame('seg', i))


palette_ade20k = np.asarray([
	[120, 120, 120],
	[180, 120, 120],
	[6, 230, 230],
	[80, 50, 50],
	[4, 200, 3],
	[120, 120, 80],
	[140, 140, 140],
	[204, 5, 255],
	[230, 230, 230],
	[4, 250, 7],
	[224, 5, 255],
	[235, 255, 7],
	[150, 5, 61],
	[120, 120, 70],
	[8, 255, 51],
	[255, 6, 82],
	[143, 255, 140],
	[204, 255, 4],
	[255, 51, 7],
	[204, 70, 3],
	[0, 102, 200],
	[61, 230, 250],
	[255, 6, 51],
	[11, 102, 255],
	[255, 7, 71],
	[255, 9, 224],
	[9, 7, 230],
	[220, 220, 220],
	[255, 9, 92],
	[112, 9, 255],
	[8, 255, 214],
	[7, 255, 224],
	[255, 184, 6],
	[10, 255, 71],
	[255, 41, 10],
	[7, 255, 255],
	[224, 255, 8],
	[102, 8, 255],
	[255, 61, 6],
	[255, 194, 7],
	[255, 122, 8],
	[0, 255, 20],
	[255, 8, 41],
	[255, 5, 153],
	[6, 51, 255],
	[235, 12, 255],
	[160, 150, 20],
	[0, 163, 255],
	[140, 140, 140],
	[250, 10, 15],
	[20, 255, 0],
	[31, 255, 0],
	[255, 31, 0],
	[255, 224, 0],
	[153, 255, 0],
	[0, 0, 255],
	[255, 71, 0],
	[0, 235, 255],
	[0, 173, 255],
	[31, 0, 255],
	[11, 200, 200],
	[255, 82, 0],
	[0, 255, 245],
	[0, 61, 255],
	[0, 255, 112],
	[0, 255, 133],
	[255, 0, 0],
	[255, 163, 0],
	[255, 102, 0],
	[194, 255, 0],
	[0, 143, 255],
	[51, 255, 0],
	[0, 82, 255],
	[0, 255, 41],
	[0, 255, 173],
	[10, 0, 255],
	[173, 255, 0],
	[0, 255, 153],
	[255, 92, 0],
	[255, 0, 255],
	[255, 0, 245],
	[255, 0, 102],
	[255, 173, 0],
	[255, 0, 20],
	[255, 184, 184],
	[0, 31, 255],
	[0, 255, 61],
	[0, 71, 255],
	[255, 0, 204],
	[0, 255, 194],
	[0, 255, 82],
	[0, 10, 255],
	[0, 112, 255],
	[51, 0, 255],
	[0, 194, 255],
	[0, 122, 255],
	[0, 255, 163],
	[255, 153, 0],
	[0, 255, 10],
	[255, 112, 0],
	[143, 255, 0],
	[82, 0, 255],
	[163, 255, 0],
	[255, 235, 0],
	[8, 184, 170],
	[133, 0, 255],
	[0, 255, 92],
	[184, 0, 255],
	[255, 0, 31],
	[0, 184, 255],
	[0, 214, 255],
	[255, 0, 112],
	[92, 255, 0],
	[0, 224, 255],
	[112, 224, 255],
	[70, 184, 160],
	[163, 0, 255],
	[153, 0, 255],
	[71, 255, 0],
	[255, 0, 163],
	[255, 204, 0],
	[255, 0, 143],
	[0, 255, 235],
	[133, 255, 0],
	[255, 0, 235],
	[245, 0, 255],
	[255, 0, 122],
	[255, 245, 0],
	[10, 190, 212],
	[214, 255, 0],
	[0, 204, 255],
	[20, 0, 255],
	[255, 255, 0],
	[0, 153, 255],
	[0, 41, 255],
	[0, 255, 204],
	[41, 0, 255],
	[41, 255, 0],
	[173, 0, 255],
	[0, 245, 255],
	[71, 0, 255],
	[122, 0, 255],
	[0, 255, 184],
	[0, 92, 255],
	[184, 255, 0],
	[0, 133, 255],
	[255, 214, 0],
	[25, 194, 194],
	[102, 255, 0],
	[92, 0, 255],
])
