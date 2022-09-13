import numpy as np

from detectron2.data import transforms as T


class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        def apply_filter(img):
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

        return T.ColorTransform(apply_filter)