import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage.filters as filters

# wczytanie obrazów
# I  = cv2.imread("I.jpg")
# J  = cv2.imread("J.jpg")
f1_original  = cv2.imread("materialy_feature_points/FeatureDetection_Input_copy.bmp")

# f2_original  = cv2.imread("fontanna2.jpg")
f1 = cv2.cvtColor(f1_original, cv2.COLOR_BGR2GRAY)
# f2 = cv2.cvtColor(f2_original, cv2.COLOR_BGR2GRAY)

# cv2.imshow("F1", f1)
# cv2.imshow("F2", f2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def harris(I_GR, size_sobel, size_gauss):
    sobelx = cv2.Sobel(I_GR, cv2.CV_32F, 1, 0, ksize=size_sobel)
    sobely = cv2.Sobel(I_GR, cv2.CV_32F, 0, 1, ksize=size_sobel)
    x2 = sobelx * sobelx
    y2 = sobely * sobely
    xy = sobelx * sobely
    gauss_x2 = cv2.GaussianBlur(x2, (size_gauss, size_gauss), 0)
    gauss_y2 = cv2.GaussianBlur(y2, (size_gauss, size_gauss), 0)
    gauss_xy = cv2.GaussianBlur(xy, (size_gauss, size_gauss), 0)
    
    det = gauss_x2*gauss_y2 - gauss_xy*gauss_xy
    trace = gauss_x2 + gauss_y2
    
    H = det - 0.05*(trace*trace)
    H = cv2.normalize(H, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return H

def find_max (image, size, threshold) : # size - maximum filter mask size
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima [diff == 0] = 0
    return np.nonzero(maxima)

def plot_im(max1, image1):
    pt_y, pt_x = max1
    arr_points = np.concatenate([pt_y[:,None], pt_x[:,None]], axis=1)
    for point in arr_points:
        print(point)
        cv2.drawMarker(image1, (point[1], point[0]), (0,0,255), markerType=cv2.MARKER_CROSS, 
        markerSize=40, thickness=2, line_type=cv2.LINE_AA)
        # image1[point[0], point[1]] = (0,0,255)
    

    cv2.imshow("feature points", image1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return image1

    # plt.figure(figsize=(12, 6))
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    # plt.plot(max1[1], max1[0], '*', color='m')
    # plt.title('Obraz 1')

# 1
# h1 = harris(f1, 7, 7)
# max1 = find_max(h1, 7, 0.55)
# new_img = plot_im(max1, f1_original)
# cv2.imwrite("harris7_7.tif", new_img)

# 2
# h1 = harris(f1, 7, 7)
# max1 = find_max(h1, 7, 0.77)
# new_img = plot_im(max1, f1_original)
# cv2.imwrite("harris7_7_2.tif", new_img)

# 3
h1 = harris(f1, 13,13)
max1 = find_max(h1, 13, 0.82)
new_img = plot_im(max1, f1_original)
cv2.imwrite("harris7_7_3.tif", new_img)
