import argparse
import json
import glob
import os
import cv2

from tqdm import tqdm
from datetime import datetime
from detectron2.structures import BoxMode


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for creating COCO annotations for COMICS.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--ds_dir", "-m", type=str, default="datasets/comics_panels_annotations",
                        help="Dataset dir")

    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Creating COCO annotations")

    now = datetime.now()
    now_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = {
        "year": now.year,
        "version": 1.0,
        "description": "COMICS panels annotations.",
        "contributor": "Atenrev",
        "url": "https://github.com/Atenrev",
        "date_created": now_formatted,
    }

    categories = [
        {
            "supercategory": "comic",
            "id": 1,
            "name": "panel",
        },
    ]

    images_paths = sorted(glob.glob(os.path.join(args.ds_dir, "data/*.jpg")))

    images = []
    annotations = []
    next_ann_id = 1

    for i, img_path in tqdm(enumerate(images_paths)):
        height, width = cv2.imread(img_path).shape[:2]

        img_basename = os.path.basename(img_path)
        image = {
            "id": i+1,
            "file_name": img_basename,
            "height": height,
            "width": width,
            "license": None,
        }

        images.append(image)
        ann_path = os.path.join(
            args.ds_dir, f"annotations/{img_basename.split('.')[0]}.txt")

        if not os.path.exists(ann_path):
            continue 

        with open(ann_path, mode="r", encoding="utf-8") as ann_f:
            for ann in ann_f:
                coords = [float(coord) for coord in ann.split()[1:]]
                coords[2] -= coords[0]
                coords[3] -= coords[1]
                segm = [
                    coords[0], coords[1],
                    coords[0] + coords[2], coords[1],
                    coords[0] + coords[2], coords[1] + coords[3],
                    coords[0], coords[1] + coords[3],
                ]
                annotations.append({
                    "id": next_ann_id,
                    "image_id": i+1,
                    "category_id": 1,
                    "bbox": coords,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [segm],
                    "area": coords[2] * coords[3],
                    "iscrowd": 0,
                })
                next_ann_id += 1

    coco_dict = {
        "info": info,
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(os.path.join(args.ds_dir, "labels.json"), "w") as f:
        json.dump(coco_dict, f)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
