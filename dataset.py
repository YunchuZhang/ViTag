import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ArucoDataset(Dataset):
	def __init__(self, input_images, dictionary, num_workers=1, transform=None):
		self.input_images = input_images
		self.dictionary = dictionary
		self.num_workers = num_workers
		self.transform = transform

	def __len__(self):
		return len(self.input_images)

	def __getitem__(self, idx):
		img_path = self.input_images[idx]
		image = cv2.imread(img_path)
		h, w = image.shape[:2]

		num_markers = random.randint(9, 12)  # Variable number of markers
		num_markers = 12
		labels = []

		elements = list(range(238, 250))
		if num_markers <= 12:
			selected_ids = random.sample(elements, num_markers)


		for marker_id in selected_ids:
			org_marker = self.generate_aruco_marker(marker_id, image.shape)
			while True:
				# scale the marker
				scale = random.uniform(0.8, 1.5)
				resize_marker = cv2.resize(org_marker, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).copy()
				marker_h, marker_w = resize_marker.shape[:2][:]
				marker, padding = self.add_padding(resize_marker.copy())
				marker = self.augment_marker(marker)
				# Check if marker can fit inside the image
				if marker.shape[0] > h or marker.shape[1] > w:
					continue  # Skip this marker if it cannot fit
				x, y = self.random_placement(image.shape, marker.shape)
				corners = np.array([
					[x, y],
					[x + marker_w, y],
					[x + marker_w, y + marker_h],
					[x, y + marker_h]
				])
				w_h, w_w = marker.shape[:2][:]
				outline_corner = np.array([
					[x-padding, y-padding],
					[x + w_w-padding, y-padding],
					[x + w_w-padding, y + w_h-padding],
					[x-padding, y + w_h-padding]
				]) 
				# Apply random rotation and warp
				# wrap_marker, warped_mask, warp_matrix = self.random_rotation_and_warp(marker)
				# corners = self.transform_corners(corners, warp_matrix)
				# outline_corner = self.transform_corners(outline_corner, warp_matrix)
				if self.is_valid(x,y,marker,w,h,outline_corner,labels):
					break


			wrap_marker = marker
			image, corners_ = self.place_marker(image, wrap_marker, outline_corner,None)

			# Append corners and id to labels
			label = corners.flatten().tolist()
			# label.append(marker_id)
			# label.append(224)
			labels.append(label)
		
		if self.transform:
			# Save original size to scale labels later
			original_size = image.shape[:2]

			# Apply transform
			image = self.transform(image)

			# Get the new size
			new_size = image.shape[1:3]

			# Scale the labels
			scale_x = new_size[1] / original_size[1]
			scale_y = new_size[0] / original_size[0]
			for i in range(len(labels)):
				for j in range(4):
					labels[i][2 * j] *= scale_x
					labels[i][2 * j + 1] *= scale_y

		# Ensure the label tensor has shape (12, 8)
		total_labels = torch.zeros((12, 8))
		total_labels[:num_markers, :] = torch.tensor(labels, dtype=torch.float32)
		total_labels = total_labels / 224.0 # normalize to 0,1
		return image.cuda(), total_labels.cuda()
	
	def is_valid(self, x, y, marker, w,h,corners, labels):
		if x < 0 or y < 0 or x + marker.shape[1] > w or y + marker.shape[0] > h:
			return False  # Skip this marker if placement is invalid
		if len(labels) == 0:
			return True
		# Compute the bounding box for the new marker
		threshold = 10
		marker_x1 = min(corners[:, 0])-threshold
		marker_y1 = min(corners[:, 1])-threshold
		marker_x2 = max(corners[:, 0])+threshold
		marker_y2 = max(corners[:, 1])+threshold

		for label in labels:
			label_corners = np.array(label[:8]).reshape(4, 2)
			label_x1 = min(label_corners[:, 0])
			label_y1 = min(label_corners[:, 1])
			label_x2 = max(label_corners[:, 0])
			label_y2 = max(label_corners[:, 1])

			# Check for overlap
			if (marker_x1 < label_x2 and marker_x2 > label_x1 and
				marker_y1 < label_y2 and marker_y2 > label_y1):
				return False

		return True    

	def generate_aruco_marker(self, idx, image_shape):
		marker_size = int(min(image_shape[0],image_shape[1])*random.uniform(0.08,0.11))
		marker_size  = max(8, marker_size)
		marker = cv2.aruco.generateImageMarker(self.dictionary, idx, marker_size)
		marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
		# marker = self.add_padding(marker, image_shape)
		return marker

	def add_padding(self, marker, padding_ratio=0.15):
		h, w = marker.shape[:2]
		padding = max(2,int(min(h, w) * padding_ratio))
		return cv2.copyMakeBorder(marker, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]),padding

	def random_placement(self, image_shape, marker_shape):
		max_x = image_shape[1] - marker_shape[1]/2
		max_y = image_shape[0] - marker_shape[0]/2
		x = random.randint(int(marker_shape[1]/2), int(max_x))
		y = random.randint(int(marker_shape[0]/2), int(max_y))
		return int(x), int(y)

	def place_marker(self, image, marker, corners, warped_mask=None):
		# Create an empty mask the size of the image
		if warped_mask is not None:
			warped_mask = np.stack([warped_mask] * 3, axis=-1)
		mask = np.zeros_like(image, dtype=np.uint8)
		# Define the marker's bounding box in the mask
		cv2.fillPoly(mask, [np.int32(corners)], (255, 255, 255))

		# Warp the marker to fit within the bounding box
		src_points = np.array([[0, 0], [marker.shape[1] - 1, 0], [marker.shape[1] - 1, marker.shape[0] - 1], [0, marker.shape[0] - 1]], dtype=np.float32)
		dst_points = corners.astype(np.float32)
		warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

		warped_marker = cv2.warpPerspective(marker, warp_matrix, (image.shape[1], image.shape[0]),flags=cv2.INTER_NEAREST)
		# warped_marker_mask = cv2.warpPerspective(warped_mask, warp_matrix, (image.shape[1], image.shape[0]),flags=cv2.INTER_NEAREST)

		# Combine the image with the warped marker using the mask
		# Combine the image with the warped marker using the mask
		combined = cv2.bitwise_or(image, mask)
		# combined = cv2.bitwise_or(combined, warped_marker_mask)
		combined[mask > 0] = warped_marker[mask > 0]

		return combined, corners
	# def place_marker(self, image, wrap_marker, corners, warped_mask):
	# 	# Create an empty mask the size of the image
	# 	mask = np.zeros_like(image, dtype=np.uint8)
		
	# 	# Define the marker's bounding box in the mask
	# 	cv2.fillPoly(mask, [np.int32(corners)], (255, 255, 255))

	# 	# Warp the marker to fit within the bounding box
	# 	src_points = np.array([[0, 0], [wrap_marker.shape[1], 0], [wrap_marker.shape[1], wrap_marker.shape[0]], [0, wrap_marker.shape[0]]], dtype=np.float32)
	# 	dst_points = corners.astype(np.float32)
	# 	warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
	# 	warped_marker = cv2.warpPerspective(wrap_marker, warp_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

	# 	# Convert warped_mask to 3-channel if not already
	# 	if len(warped_mask.shape) == 2:
	# 		warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)

	# 	# Ensure warped_mask has same size as warped_marker
	# 	warped_mask = cv2.resize(warped_mask, (warped_marker.shape[1], warped_marker.shape[0]))

	# 	# Combine the image with the warped marker using the mask
	# 	combined = np.where(mask > 0, warped_marker, image)

	# 	return combined, corners


	def augment_marker(self, marker):
		
		marker = self.apply_random_grayscale(marker)
		marker = self.apply_gaussian_blur(marker)
		marker = self.apply_gaussian_noise(marker)
		marker = self.apply_motion_blur(marker)
		marker = self.add_color_noise(marker)
		marker = self.reduce_brightness(marker,factor=0.5)
		marker = self.apply_spotlighting_effect(marker)
		return marker

	def apply_random_grayscale(self, image):
		if random.random() < 0.5:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		return image

	def apply_spotlighting_effect(self, image,strength=1.05):
		if np.random.rand() > 0.9:
			rows, cols = image.shape[:2]
			center_x = np.random.randint(0, cols)
			center_y = np.random.randint(0, rows)
			radius = min(rows, cols)
			Y, X = np.ogrid[:rows, :cols]
			dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
			circular_gradient = np.clip((radius - dist_from_center) / radius, 0, 1)

			spotlight = np.dstack([circular_gradient] * 3) * strength

			image = np.clip(image * spotlight, 0, 255).astype(np.uint8)
		return image

	def apply_gaussian_blur(self, image):
		if random.random() > 0.6:
			ksize = random.choice([(3, 3), (5, 5)])
			image = cv2.GaussianBlur(image, ksize, 0)
		return image

	def apply_gaussian_noise(self, image):
		if random.random() < 0.3:
			noise = np.random.normal(0, 0.4, image.shape).astype(np.uint8)
			image = cv2.add(image, noise)
		return image

	def apply_motion_blur(self, image):
		if random.random() < 0.4:
			ksize = random.randint(2,3)
			kernel = np.zeros((ksize, ksize))
			xs, ys = random.choice([(1, 0), (0, 1), (1, 1)])
			cv2.line(kernel, (0, 0), (ksize * xs, ksize * ys), 1, thickness=1)
			kernel = kernel / ksize
			image = cv2.filter2D(image, -1, kernel)
		return image
	
	def add_color_noise(self, image, alpha=0.25):
		if random.random() < 0.5:
			colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (123,123,123)]
			color = random.choice(colors)
			overlay = np.full(image.shape, color, dtype=np.uint8)
			image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
		return image

	def reduce_brightness(self, image, factor=0.5):
		factor = max(np.random.rand(),factor)
		return np.clip(image * factor, 0, 255).astype(np.uint8)
	

	
	def random_rotation_and_warp(self, image):
		angle = random.uniform(-90, 90)
		(h, w) = image.shape[:2]
		(cX, cY) = (w / 2, h / 2)

		# Calculate rotation matrix
		M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])

		# Compute the new bounding dimensions of the image
		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))

		# Adjust the rotation matrix to take into account translation
		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY
		rotation_matrix = M.copy()

		# Perform the actual rotation and return the image
		rotated_image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC)

		# Create a mask for the rotated image
		mask = np.ones((h, w), dtype=np.uint8) * 255
		rotated_mask = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_CUBIC)

		# Random perspective transform
		src_points = np.array([
			[0, 0],
			[w - 1, 0],
			[w - 1, h - 1],
			[0, h - 1]
		], dtype=np.float32)

		max_warp = 0.02
		dst_points = src_points + np.random.uniform(-max_warp, max_warp, src_points.shape).astype(np.float32) * np.array([w, h])
		warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points.astype(np.float32))

		# Adjust src_points for the new dimensions after rotation
		rotated_src_points = cv2.transform(np.array([src_points]), rotation_matrix)[0]

		# Apply the perspective transform to the rotated image and mask
		warped_image = cv2.warpPerspective(rotated_image, warp_matrix, (nW, nH), flags=cv2.INTER_CUBIC)
		warped_mask = cv2.warpPerspective(rotated_mask, warp_matrix, (nW, nH), flags=cv2.INTER_CUBIC)

		return warped_image, warped_mask, warp_matrix



	def transform_corners(self, corners, warp_matrix):
		corners = np.hstack((corners, np.ones((4, 1))))
		transformed_corners = warp_matrix @ corners.T
		transformed_corners = transformed_corners[:2] / transformed_corners[2]
		return transformed_corners.T

	def save_and_visualize(self, image, labels, save_path,vis=False):
		plt.figure(figsize=(20, 20))
		if vis:
			plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		else:
			plt.imshow(image[:,:,::-1])
		
		for label in labels:
			corners = np.array(label[:8]).reshape(-1, 2)
			plt.plot([corners[0][0], corners[1][0]], [corners[0][1], corners[1][1]], 'r-',linewidth=2)
			plt.plot([corners[1][0], corners[2][0]], [corners[1][1], corners[2][1]], 'r-',linewidth=2)
			plt.plot([corners[2][0], corners[3][0]], [corners[2][1], corners[3][1]], 'r-',linewidth=2)
			plt.plot([corners[3][0], corners[0][0]], [corners[3][1], corners[0][1]], 'r-',linewidth=2)
			plt.scatter(corners[:,0],corners[:,1], c='m',marker='o',s=180)
			#plt.text(corners[0][0], corners[0][1], str(int(label[8])), color='blue', fontsize=12, weight='bold')
		
		plt.axis('off')
		plt.savefig(save_path)
		
		if vis:
			plt.show()
		plt.close()
if __name__ == '__main__':
	input_images = ["/home/yunchuz/deeptag/dataset_org/0.jpg", "/home/yunchuz/deeptag/dataset_org/0.jpg"]  # List of input image paths
	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)

	# Define custom transforms using torchvision.transforms.v2
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# Create dataset instance
	dataset = ArucoDataset(input_images, aruco_dict, transform=transform)

	# Create DataLoader instance
	dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
	for batch_idx, (images, labels) in enumerate(dataloader):
		batch_size = images.size(0)
		
		# Iterate through each image in the batch
		for idx in range(batch_size):
			# import pdb;pdb.set_trace()
			image = images[idx].cpu().numpy().transpose(1, 2, 0) 
			batch_labels = labels[idx].cpu().numpy()

			# Example visualization for each image in the batch
			dataset.save_and_visualize(image, batch_labels*224, f'batch{batch_idx}_image{idx}_visualization.png',vis=True)