import numpy as np
import cv2

from detectron2.data import transforms as T


class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        def palette_filter(img):
            img = np.array(img)
            r = np.random.rand(2)
            r[0] = r[0] * 0.2 + 0.8
            r[1] = r[1] * 10
            img[:,:,0] = img[:,:,0] * r[0] + r[1]
            r = np.random.rand(2)
            r[0] = r[0] * 0.2 + 0.8
            r[1] = r[1] * 10
            img[:,:,1] = img[:,:,1] * r[0] + r[1]
            r = np.random.rand(2)
            r[0] = r[0] * 0.2 + 0.8
            r[1] = r[1] * 10
            img[:,:,2] = img[:,:,2] * r[0] + r[1]
            return img
            # image_bw = to_bw(image)
            # return image_bw

        def canny_filter(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            th1 = np.random.randint(25, 185)
            th2 = max(th1 + np.random.randint(50, 150), 235)
            edges = 255 - cv2.Canny(image=blurred, threshold1=th1, threshold2=th2)
            img = np.array(img)
            img[:,:,0] = edges
            img[:,:,1] = edges
            img[:,:,2] = edges
            return img

        if np.random.random() < 0.9:
            return T.ColorTransform(palette_filter)
        else:
            return T.ColorTransform(canny_filter)