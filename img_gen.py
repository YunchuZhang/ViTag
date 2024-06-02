import cv2
import glob
import numpy as np

def add_spotlight_effect(image, radius=None, strength=1.15):
    if np.random.rand() > 0.8:
        rows, cols = image.shape[:2]

        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)

        if radius is None:
            radius = min(rows, cols)

        Y, X = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        circular_gradient = np.clip((radius - dist_from_center) / radius, 0, 1)

        spotlight = np.dstack([circular_gradient] * 3) * strength

        image = np.clip(image * spotlight, 0, 255).astype(np.uint8)
    return image

def add_random_grayscale(image):
    if np.random.rand() > 0.7:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def add_gaussian_blur(image, ksize=(5, 5)):
    if np.random.rand() > 0.5:
        image = cv2.GaussianBlur(image, ksize, 0)
    return image

def add_gaussian_noise(image, mean=0, sigma=10):
    if np.random.rand() > 0.5:
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
        image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return image

def add_motion_blur(image, size=10):
    if np.random.rand() > 0.5:
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image

def reduce_brightness(image, factor=0.5):
    factor = max(np.random.rand(),0.5)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def process_image(image_path, output_path):

    image = cv2.imread(image_path)
    if image is None:
        return False


    image = add_random_grayscale(image)
    image = add_gaussian_blur(image)
    image = add_gaussian_noise(image)
    image = add_motion_blur(image)
    image = reduce_brightness(image)
    image = add_spotlight_effect(image)

    cv2.imwrite(output_path, image)
    return True

# folder_path = '/home/yunchuz/deeptag/dataset/'
# output_folder = '/home/yunchuz/deeptag/dataset_org/'
folder_path = '/home/yunchuz/deeptag/dataset_org/'
output_folder = '/home/yunchuz/deeptag/dataset_org_noise/'
files_list = glob.glob(f"{folder_path}/**", recursive=True)
print(len(files_list))


cnt = 0
# for i in range(19):
for i in range(len(files_list)):
    f = files_list[i]
    if 'jpg' in f or 'png' in f:
        f_n = f.split('/')[-1].split('.')[0]
        print(f"cur file name: {f_n}")
        for j in range(5):
            # output_path = f'{output_folder}{f_n}_noise.png'
            output_path = f'{output_folder}{cnt}_noise_{j}.jpg'
            is_valid = process_image(f, output_path)
            if is_valid:
                cnt += 1