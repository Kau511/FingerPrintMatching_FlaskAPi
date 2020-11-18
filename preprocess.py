import sys
import numpy as np
import cv2
# from zhangsuen import ZhangSuen
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.signal as signal
import helper
from ridge_filter import ridge_filter


# @jit(nopython=True)
def preprocess(image):

	# Normalize and Segment
	image, mask = helper.segment(image)
	# Get Orientations
	orientations = helper.getOrientations(image)
	# Get Frequencies
	frequencies = helper.getFrequencies(image,orientations)
	freq = frequencies*mask

	# Gabor Filter
	image = ridge_filter(image, orientations, freq, 0.7, 0.7);  
	image = (image < -3).astype(int)

	return image*255, mask, orientations



if __name__ == "__main__":
	file_name = sys.argv[1]
	img = cv2.imread(file_name, 0)
	image = img.astype(float)

	# Normalize and Segment
	image, mask = helper.segment(image)
	# Get Orientations
	orientations = helper.getOrientations(image)
	# Get Frequencies
	frequencies = helper.getFrequencies(image,orientations)
	freq = frequencies*mask

	# Gabor Filter
	image = ridge_filter(image, orientations, freq, 0.65, 0.65);  

	image = (image < -3).astype(int)

	cv2.imwrite('enh.jpg', image*255)
