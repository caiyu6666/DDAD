import pydicom
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

in_dir = "stage_2_train_images"  # Your path to original data (.dcm or .dicom)
out_dir = "train_png_512"  # Your output path for preprocessed data (.png)
img_size = 512  # image size of png


if __name__ == "__main__":
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for img_name in tqdm(os.listdir(in_dir)):
		dicom = pydicom.read_file(os.path.join(in_dir, img_name))
		img = dicom.pixel_array
		if dicom.PhotometricInterpretation == "MONOCHROME1":
			img = np.max(img) - img

		if img.dtype != np.uint8:
			img = ((img - np.min(img)) * 1.0 / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

		img = Image.fromarray(img).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
		img.save(os.path.join(out_dir, img_name.split(".")[0] + ".png"))
