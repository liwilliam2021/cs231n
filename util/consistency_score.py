# CITING:
# Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C.: SUPER-RESOLUTION 
# OF MULTISPECTRAL SATELLITE IMAGES USING CONVOLUTIONAL NEURAL NETWORKS, ISPRS 
# Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-1-2020, 33–40, 
# https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020, 2020.

from image_similarity_measures.quality_metrics import rmse, sam, structural_similarity
import cv2


# print ("Beg eval")
# value1 = evaluation(org_img_path="tempImages/image0.png", 
#            pred_img_path="tempImages/image1.png", 
#            metrics=["rmse"])
# print (value1)
# image0 = data_img = cv2.imread("tempImages/image0.png")
# image1 = data_img = cv2.imread("tempImages/image1.png")
# value0 = rmse(image0, image1)
# print (value0)
# print ("End eval")

images = []
for i in range (5):
  images.append(cv2.imread(f"tempImages/image{i}.png"))
for i in range (0, 5):
  for j in range (0, 5):
    print ((i,j))
    print(rmse(images[i], images[j]))