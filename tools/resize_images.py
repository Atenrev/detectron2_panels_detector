import argparse
import glob
import os
import cv2

from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for resizing a dataset.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--ds_dir", "-m", type=str, default="datasets/eBDtheque_database_v3_99",
                        help="Dataset dir")

    return parser.parse_args()


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def main(args: argparse.Namespace):
    images_paths = sorted(glob.glob(os.path.join(args.ds_dir, "data_org/*.jpg")))

    for i, img_path in tqdm(enumerate(images_paths)):
        img = cv2.imread(img_path)
        img = image_resize(img, height=1200, width=800)
        img_basename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(f"{args.ds_dir}/data/", img_basename), img)
        

if __name__ == "__main__":
    args = _parse_args()
    main(args)
