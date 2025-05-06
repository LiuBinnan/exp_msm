import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import cv2
from pytorch_msssim import ssim

from model import ContentEmbedding, StyleEmbedding, MaskedToken, TransformerEncoder, StylizedEnhancer
from utils import AverageMeter, reverse_normalize, rgb2lab
from dataset.loader import Dataset2Train, Dataset2ValidPref, Dataset2ValidUnseen

from pyiqa import create_metric 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import calculate_psnr_pt, calculate_ssim_pt, calculate_eab_pt
import logging

groups = [
    "grp_0_subgrp_91",
    "grp_11_subgrp_20",
    "grp_18_subgrp_111",
    "grp_4_subgrp_6",
    "grp_8_subgrp_85",
	]

parser = argparse.ArgumentParser()
parser.add_argument('--v_batch_size', default=10, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='/media/liubinnan/新加卷/checkpoint/masked_style_modeling/pretrained_models', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/home/liubinnan/test_bench_for_MSM', type=str, help='path to dataset')
parser.add_argument('--spatial_size', default=2, type=int)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--reference_split', default='/home/liubinnan/code/reference_split.json', type=str)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./test_result.log"),  # 写入文件
        logging.StreamHandler()               # 输出到终端
    ]
)

def test(test_unseen_loader, test_preferred_loader,
	transformer_encoder, masked_token, style_embedding, content_embedding, stylized_enhancer,
	style_dirs, num_pref_images, group_id):

	PSNR = [AverageMeter() for i in range(len(style_dirs))]
	SSIM = [AverageMeter() for i in range(len(style_dirs))]
	LPIPS = [AverageMeter() for i in range(len(style_dirs))]
	DELTAab = [AverageMeter() for i in range(len(style_dirs))]

	torch.cuda.empty_cache()

	transformer_encoder.eval()
	masked_token.eval()
	content_embedding.eval()
	style_embedding.eval()
	stylized_enhancer.eval()

	pca_components = np.load(os.path.join(args.save_dir, "pca_components.npy"))
	pca_mean = np.load(os.path.join(args.save_dir, "pca_mean.npy"))
	pca_components = torch.from_numpy(pca_components.astype('float32')).cuda(non_blocking=True)
	pca_std = (pca_components**2).sum(dim=1)**0.5
	pca_components = pca_components / pca_std[:,None]
	pca_mean = torch.from_numpy(pca_mean.astype('float32')).cuda(non_blocking=True)

	counts = torch.zeros((len(style_dirs), 1)).cuda(non_blocking=True)

	preferred_y_256_set = torch.zeros((len(style_dirs), num_pref_images, 3, 256, 256)).cuda(non_blocking=True)
	preferred_x_256_set = torch.zeros((len(style_dirs), num_pref_images, 3, 256, 256)).cuda(non_blocking=True)
	preferred_s_set = torch.zeros((len(style_dirs), num_pref_images, 512)).cuda(non_blocking=True)
	preferred_c_set = torch.zeros((len(style_dirs), num_pref_images, 512)).cuda(non_blocking=True)

	with torch.no_grad():
		for (y_256, x_256, style_id, img_name) in tqdm(test_preferred_loader, file=sys.stdout):
			y_256 = y_256.cuda(non_blocking=True)
			x_256 = x_256.cuda(non_blocking=True)

			for i in range(y_256.shape[0]):
				counts[style_id[i], 0] += 1
				preferred_y_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = y_256[i]
				preferred_x_256_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = x_256[i]
				preferred_s_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = style_embedding(x_256[i:i+1], y_256[i:i+1])[0]
				preferred_c_set[style_id[i], counts[style_id[i], 0].to(torch.int64)-1] = content_embedding(x_256[i:i+1])[0]

				# if args.save_images:
				# 	os.symlink("../../../"+img_name[i],
				# 		"test_results/{0}/{3}/{1}/preferred/{2}".format(style_dirs[style_id[i].cpu()], num_pref_images, os.path.basename(img_name[i]), group_id))

		if args.save_images:
			for i in range(len(style_dirs)):
				# preferred_image_matrix = np.zeros((128*(num_pref_images//5), 128*5, 3), dtype=np.uint8)
				preferred_images = np.zeros((num_pref_images, 128, 128, 3), dtype=np.uint8)
				for j in range(num_pref_images):
					preferred_y_256_np = reverse_normalize(preferred_y_256_set[i, j]).cpu().numpy()[0]
					preferred_y_256_np = np.clip(preferred_y_256_np.transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					preferred_images[j] = cv2.resize(preferred_y_256_np, (128, 128))
					# cv2.imwrite("test_results/{0}/{2}/{1}/preferred/{3}".format(style_dirs[i], num_pref_images, group_id, img_name[j]), preferred_y_256_np)
					# preferred_image_matrix[128*(j//5):128*(j//5+1), 128*(j%5):128*(j%5+1)] = cv2.resize(preferred_y_256_np, (128, 128))
				# cv2.imwrite("test_results/{0}/{2}/{1}/preferred/reference.jpg".format(style_dirs[i], num_pref_images, group_id), preferred_images.reshape(128, -1, 3))



		for (x, y, x_256, style_id, original_shape, img_name, my_style_dirs) in tqdm(test_unseen_loader, file=sys.stdout):
			x = x.cuda(non_blocking=True)
			y = y.cuda(non_blocking=True)
			style_id = style_id.cuda(non_blocking=True)

			preferred_c = preferred_c_set[style_id.argmax(dim=1)]
			preferred_s = preferred_s_set[style_id.argmax(dim=1)]
			batch_size = preferred_c.shape[0]
			num_pref_images = preferred_c.shape[1]

			c = content_embedding(x_256)

			concat_s = torch.cat([preferred_s, masked_token(torch.ones_like(c[:,0:1]))[:, None]], dim=1)
			concat_c = torch.cat([preferred_c, c[:, None]], dim=1)
			A = torch.cat([torch.zeros_like(concat_s), torch.zeros_like(concat_s)], dim=2)
			A[:,:,::2] = concat_s
			A[:,:,1::2] = concat_c

			predicted_s = transformer_encoder(A)
			predicted_s = torch.mm(predicted_s * pca_std[None], pca_components) + pca_mean[None]

			delta = stylized_enhancer(x, predicted_s.reshape(list(predicted_s.shape)+[1,1]))
			predicted_y = delta + x

			predicted_y_rgb = reverse_normalize(predicted_y)
			y_rgb = reverse_normalize(y)
			# predicted_y_lab = rgb2lab(predicted_y_rgb)
			# y_lab = rgb2lab(y_rgb)
			predicted_y_rgb_ = (predicted_y_rgb.clamp_(0, 1) * 255).round_() / 255.
			y_rgb_ = (y_rgb.clamp_(0, 1) * 255).round_() / 255.

			# mse = F.mse_loss(predicted_y_rgb, y_rgb, reduction='none').mean((1, 2, 3))

			psnr_val = calculate_psnr_pt(predicted_y_rgb, y_rgb, 0)
			ssim_val = calculate_ssim_pt(predicted_y_rgb, y_rgb, 0)
			lpips_ = create_metric("lpips", device="cuda")
			lpips_val = lpips_(predicted_y_rgb_, y_rgb_)
			eab_val = calculate_eab_pt(predicted_y_rgb, y_rgb)

			# PSNR[style_id[0].argmax().item()].update(psnr_val)
			# SSIM[style_id[0].argmax().item()].update(ssim_val)
			# LPIPS[style_id[0].argmax().item()].update(lpips_val)
			# DELTAab[style_id[0].argmax().item()].update(eab_val)

			for i in range(batch_size):
				# psnr = 10 * torch.log10(1 / mse[i])
				PSNR[style_id[i].argmax().item()].update(psnr_val[i].item(), 1)
				DELTAab[style_id[i].argmax().item()].update(eab_val[i].item(), 1)

				# _, _, H, W = y_rgb.size()
				# down_ratio = max(1, round(min(H, W) / 256))
				# ssim_score = ssim(F.adaptive_avg_pool2d(predicted_y_rgb[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
				# 				F.adaptive_avg_pool2d(y_rgb[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
				# 				data_range=1, size_average=False).item()
				SSIM[style_id[i].argmax().item()].update(ssim_val[i].item(), 1)
				LPIPS[style_id[i].argmax().item()].update(lpips_val[i].item(), 1)

			if args.save_images:
				predicted_y_np = predicted_y_rgb.cpu().numpy()
				y_np = reverse_normalize(y).cpu().numpy()
				x_np = reverse_normalize(x).cpu().numpy()
				for i in range(batch_size):
					predicted_y_np_i = np.clip(predicted_y_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					predicted_y_np_i = cv2.resize(predicted_y_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)
					y_np_i = np.clip(y_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					y_np_i = cv2.resize(y_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)
					x_np_i = np.clip(x_np[i].transpose((1,2,0)) * 255, 0, 255).astype(np.uint8)[:,:,::-1]
					x_np_i = cv2.resize(x_np_i, original_shape[i][:2].tolist()[::-1], interpolation=cv2.INTER_LINEAR)

					os.makedirs("./result_{0}/{1}/{2}".format(groups[group_id], num_pref_images, my_style_dirs[i]), exist_ok=True)
					os.makedirs("./vis_{0}/{1}/{2}".format(groups[group_id], num_pref_images, my_style_dirs[i]), exist_ok=True)
					output_name_ = "./result_{0}/{1}/{3}/{2}".format(groups[group_id], num_pref_images, img_name[i], my_style_dirs[i])
					output_name = "./vis_{0}/{1}/{3}/{2}".format(groups[group_id], num_pref_images, img_name[i], my_style_dirs[i])
					cv2.imwrite(output_name, np.hstack([x_np_i, predicted_y_np_i, y_np_i]))
					cv2.imwrite(output_name_, predicted_y_np_i)


	logging.info("group:{4} PSNR:{0:.3f} SSIM:{1:.3f} LPIPS:{3:.3f} DELTAab:{2:.3f}".format(np.array(list(map(lambda x: x.avg, PSNR))).mean(),
												np.array(list(map(lambda x: x.avg, SSIM))).mean(),
												np.array(list(map(lambda x: x.avg, DELTAab))).mean(),
												np.array(list(map(lambda x: x.avg, LPIPS))).mean(),
												group_id))

	# print("PSNR:{0:.3f} SSIM:{1:.3f} LPIPS:{3:.3f} DELTAab:{2:.3f}".format(np.array(list(map(lambda x: x.avg, PSNR))).mean(),
	# 											np.array(list(map(lambda x: x.avg, SSIM))).mean(),
	# 											np.array(list(map(lambda x: x.avg, DELTAab))).mean(),
	# 											np.array(list(map(lambda x: x.avg, LPIPS))).mean()))
	return list(map(lambda x: x.avg, PSNR)), list(map(lambda x: x.avg, SSIM)), list(map(lambda x: x.avg, LPIPS)), list(map(lambda x: x.avg, DELTAab))


if __name__ == '__main__':
	transformer_encoder = TransformerEncoder()
	transformer_encoder = nn.DataParallel(transformer_encoder).cuda()
	transformer_encoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'transformer_encoder.pth.tar'))['state_dict'])

	masked_token = MaskedToken()
	masked_token = nn.DataParallel(masked_token).cuda()
	masked_token.load_state_dict(torch.load(os.path.join(args.save_dir, 'masked_token.pth.tar'))['state_dict'])

	style_embedding = StyleEmbedding()
	style_embedding = nn.DataParallel(style_embedding).cuda()
	style_embedding.load_state_dict(torch.load(os.path.join(args.save_dir, 'style_embedding.pth.tar'))['state_dict'])

	content_embedding = ContentEmbedding(args.spatial_size)
	content_embedding = nn.DataParallel(content_embedding).cuda()
	content_embedding.load_state_dict(torch.load(os.path.join(args.save_dir, 'content_embedding.pth.tar'))['state_dict'])

	stylized_enhancer = StylizedEnhancer()
	stylized_enhancer = nn.DataParallel(stylized_enhancer).cuda()
	stylized_enhancer.load_state_dict(torch.load(os.path.join(args.save_dir, 'stylized_enhancer.pth.tar'))['state_dict'])


	# test_dataset = Dataset2ValidUnseen('test', args)
	# style_dirs = list(map(lambda x: os.path.basename(x), test_dataset.style_dirs))
	style_dirs = ["1", "2", "3", "4", "5"]
	group_num = 5
	groups = [
    "grp_0_subgrp_91",
    "grp_11_subgrp_20",
    "grp_18_subgrp_111",
    "grp_4_subgrp_6",
    "grp_8_subgrp_85",
	]

	if not args.save_images:
		num_pref_images_list = [3, 5, 10]
	else:
		num_pref_images_list = [3, 5, 10]

	# PSNRresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))
	# SSIMresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))
	# DELTAabresults = np.zeros((10, len(style_dirs), len(num_pref_images_list)))
	PSNRresults = np.zeros((group_num, len(style_dirs), len(num_pref_images_list)))
	SSIMresults = np.zeros((group_num, len(style_dirs), len(num_pref_images_list)))
	LPIPSresults = np.zeros((group_num, len(style_dirs), len(num_pref_images_list)))
	DELTAabresults = np.zeros((group_num, len(style_dirs), len(num_pref_images_list)))

	for k, num_pref_images in enumerate(num_pref_images_list):
		for i in range(group_num): # 重复计算 10 次 -> 计算不同 group
			os.makedirs("./result_{0}/{1}".format(groups[i], num_pref_images), exist_ok=True)
			os.makedirs("./vis_{0}/{1}".format(groups[i], num_pref_images), exist_ok=True)
			print("[Number of preferred images: {0}] {1} / {2}".format(num_pref_images, i + 1, group_num))
			args.data_dir = os.path.join("/home/liubinnan/test_bench_for_MSM", groups[i])
			test_ref_dataset = Dataset2ValidPref('test_ref', args, num_pref_images, start_idx=i)
			test_preferred_loader = DataLoader(test_ref_dataset,
		                            batch_size=args.v_batch_size,
		                            num_workers=5,
		                            pin_memory=True)

			test_dataset = Dataset2ValidUnseen('test', args, start_idx=i)
			test_unseen_loader = DataLoader(test_dataset,
		                            batch_size=args.v_batch_size,
		                            num_workers=5,
		                            pin_memory=True)
			
			# if args.save_images:
			# 	for style_dir in style_dirs:
			# 		os.makedirs("test_results/{0}/{2}/{1}/preferred/".format(style_dir, num_pref_images, i))
			# 		os.makedirs("test_results/{0}/{2}/{1}/output/".format(style_dir, num_pref_images, i))

			PSNR, SSIM, LPIPS, DELTAab = test(test_unseen_loader, test_preferred_loader,
				transformer_encoder, masked_token, style_embedding, content_embedding, stylized_enhancer,
				style_dirs, num_pref_images, group_id=i)

			for j, dir in enumerate(style_dirs):
				PSNRresults[i, j, k] = PSNR[j]
				SSIMresults[i, j, k] = SSIM[j]
				LPIPSresults[i, j, k] = LPIPS[j]
				DELTAabresults[i, j, k] = DELTAab[j]

	if args.save_images:
		print("Results are saved in test_results/")

	print("[Results]")
	for k, num_pref_images in enumerate(num_pref_images_list):
		logging.info("[Number of preferred images: {0}]".format(num_pref_images))
		logging.info("PSNR: {0:.3f}@{1:.3f}".format(PSNRresults[:,:,k].mean(axis=1).mean(), PSNRresults[:,:,k].mean(axis=1).std()))
		logging.info("SSIM: {0:.3f}±{1:.3f}".format(SSIMresults[:,:,k].mean(axis=1).mean(), SSIMresults[:,:,k].mean(axis=1).std()))
		logging.info("LPIPS: {0:.3f}±{1:.3f}".format(LPIPSresults[:,:,k].mean(axis=1).mean(), LPIPSresults[:,:,k].mean(axis=1).std()))
		logging.info("DELTAab: {0:.3f}±{1:.3f}".format(DELTAabresults[:,:,k].mean(axis=1).mean(), DELTAabresults[:,:,k].mean(axis=1).std()))
		# print("[Number of preferred images: {0}]".format(num_pref_images))
		# print("PSNR: {0:.3f}@{1:.3f}".format(PSNRresults[:,:,k].mean(axis=1).mean(), PSNRresults[:,:,k].mean(axis=1).std()))
		# print("SSIM: {0:.3f}±{1:.3f}".format(SSIMresults[:,:,k].mean(axis=1).mean(), SSIMresults[:,:,k].mean(axis=1).std()))
		# print("LPIPS: {0:.3f}±{1:.3f}".format(LPIPSresults[:,:,k].mean(axis=1).mean(), LPIPSresults[:,:,k].mean(axis=1).std()))
		# print("DELTAab: {0:.3f}±{1:.3f}".format(DELTAabresults[:,:,k].mean(axis=1).mean(), DELTAabresults[:,:,k].mean(axis=1).std()))
	
	# test_dataset = Dataset2ValidUnseen('test', args)
	# style_dirs = list(map(lambda x: os.path.basename(x), test_dataset.style_dirs))
	# style_dirs = ["target_a"]

	# num_pref_images_list = [1, 2, 3, 4, 5, 6]

	# PSNRresults = np.zeros((21, len(style_dirs), len(num_pref_images_list)))
	# SSIMresults = np.zeros((21, len(style_dirs), len(num_pref_images_list)))
	# DELTAabresults = np.zeros((21, len(style_dirs), len(num_pref_images_list)))

	# for k, num_pref_images in enumerate(num_pref_images_list):
	# 	for i in range(21): # 重复计算 10 次 -> 计算不同 group 
	# 		print("[Number of preferred images: {0}] {1} / 21".format(num_pref_images, i))
	# 		# test_ref_dataset = Dataset2ValidPref('test_ref', args, num_pref_images, start_idx=i)
	# 		test_ref_dataset = Dataset2ValidPref_PPR10K_group(args, num_pref_images, i)
	# 		test_preferred_loader = DataLoader(test_ref_dataset,
	# 	                            batch_size=args.v_batch_size,
	# 	                            num_workers=5,
	# 	                            pin_memory=True)

	# 		# test_dataset = Dataset2ValidUnseen('test', args)
	# 		test_dataset = Dataset2ValidUnseen_PPR10K_group(args, i)
	# 		test_unseen_loader = DataLoader(test_dataset,
	# 	                            batch_size=args.v_batch_size,
	# 	                            num_workers=5,
	# 	                            pin_memory=True)

	# 		if args.save_images:
	# 			for style_dir in style_dirs:
	# 				os.makedirs("test_results/{0}/{1}/preferred/".format(style_dir, num_pref_images), exist_ok=True)
	# 				os.makedirs("test_results/{0}/{1}/output/".format(style_dir, num_pref_images), exist_ok=True)

	# 		PSNR, SSIM, DELTAab = test(test_unseen_loader, test_preferred_loader,
	# 			transformer_encoder, masked_token, style_embedding, content_embedding, stylized_enhancer,
	# 			style_dirs, num_pref_images)

	# 		for j, dir in enumerate(style_dirs):
	# 			PSNRresults[i, j, k] = PSNR[j]
	# 			SSIMresults[i, j, k] = SSIM[j]
	# 			DELTAabresults[i, j, k] = DELTAab[j]

	# print("[Results]")
	# for k, num_pref_images in enumerate(num_pref_images_list):
	# 	print("[Number of preferred images: {0}]".format(num_pref_images))
	# 	print("PSNR: {0:.2f}@{1:.2f}".format(PSNRresults[:,:,k].mean(axis=1).mean(), PSNRresults[:,:,k].mean(axis=1).std()))
	# 	print("SSIM: {0:.3f}±{1:.3f}".format(SSIMresults[:,:,k].mean(axis=1).mean(), SSIMresults[:,:,k].mean(axis=1).std()))
	# 	print("DELTAab: {0:.2f}±{1:.2f}".format(DELTAabresults[:,:,k].mean(axis=1).mean(), DELTAabresults[:,:,k].mean(axis=1).std()))
