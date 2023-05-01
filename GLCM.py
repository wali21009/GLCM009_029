import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Load image
img = cv2.imread('image/toiletpaper.jpg', cv2.IMREAD_GRAYSCALE)

# Define GLCM properties
d = 1
theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Compute GLCM
glcm = graycomatrix(img, distances=[d], angles=theta, levels=256, symmetric=True, normed=True)

# Compute GLCM properties
contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
correlation = graycoprops(glcm, 'correlation')
energy = graycoprops(glcm, 'energy')

# Print results
print('Contrast: ', contrast[0][0])
print('Homogeneity: ', homogeneity[0][0])
print('Correlation: ', correlation[0][0])
print('Energy: ', energy[0][0])