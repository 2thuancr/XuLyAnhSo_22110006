import os
import cv2
import numpy as np
path = "../dataset_traicaynew_scratch"
path_out = "../dataset_traicaynew/Newfolder"
dir_list = os.listdir(path)

for file_name in dir_list:
    full_name = path + '/' + file_name
    print(full_name)

dem = 1
for file_name in dir_list:
    full_name = path + '/' + file_name
    full_name_out = '%s/%03d.jpg' % (path_out, dem)
    dem = dem + 1

    imgin = cv2.imread(full_name, cv2.IMREAD_COLOR)
    M, N, C = imgin.shape
    print(M, N, C)
    if M > N:
        imgout = np.zeros((M,M,C),np.uint8) + 255
        imgout[:M,:N,:] = imgin
        imgout = cv2.resize(imgout, (416,416))
    elif M < N:
        imgout = np.zeros((N,N,C),np.uint8) + 255
        imgout[:M,:N,:] = imgin
        imgout = cv2.resize(imgout, (416,416))
    else:
        imgout = cv2.resize(imgin, (416,416))
    cv2.imwrite(full_name_out, imgout)
