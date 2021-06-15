# import numpy as np
# import random
# import cv2

# def sp_noise(image,prob):
#     '''
#     Add salt and pepper noise to image
#     prob: Probability of the noise
#     '''
#     output = np.zeros(image.shape,np.uint8)
#     thres = 1 - prob
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output
# if __name__ == "__main__":
#     img_path = input("Enter img path: ")
#     image = cv2.imread(img_path,0) # Only for grayscale image
#     noise_img = sp_noise(image,0.02)
#     cv2.imwrite('sp_noise1.jpg', noise_img)


# import numpy as np
# import cv2
# img_path = input("Enter img path: ")
# img = cv2.imread(img_path)
# mean = 0
# var = 10
# sigma = var ** 0.5
# gaussian = np.random.normal(mean, sigma, (224, 224)) #  np.zeros((224, 224), np.float32)
#
# noisy_image = np.zeros(img.shape, np.float32)
#
# if len(img.shape) == 2:
#     noisy_image = img + gaussian
# else:
#     noisy_image[:, :, 0] = img[:, :, 0] + gaussian
#     noisy_image[:, :, 1] = img[:, :, 1] + gaussian
#     noisy_image[:, :, 2] = img[:, :, 2] + gaussian
#
# cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
# noisy_image = noisy_image.astype(np.uint8)
#
# cv2.imshow("img", img)
# cv2.imshow("gaussian", gaussian)
# cv2.imshow("noisy", noisy_image)
#
# cv2.waitKey(0)

import numpy as np
from skimage.util import random_noise
import cv2

# Load the image
img_path = input("Enter img path: ")
img = cv2.imread(img_path)

# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='s&p',amount=0.05)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')

# Display the noise image
cv2.imshow('blur',noise_img)
cv2.waitKey(0)
