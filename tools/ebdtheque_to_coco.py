import argparse
import json
import glob
import os
import cv2

from tqdm import tqdm
from datetime import datetime
from detectron2.structures import BoxMode
from xml.dom import minidom


PANEL_ID = 1
BUBBLE_ID = 2
CHARACTER_ID = 3


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for creating COCO annotations for eBDtheque.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--ds_dir", "-m", type=str, default="datasets/eBDtheque_database_v3",
                        help="Dataset dir")

    return parser.parse_args()


def main(args: argparse.Namespace):
    print("Creating COCO annotations")

    now = datetime.now()
    now_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = {
        "year": now.year,
        "version": 1.0,
        "description": "eBDtheque panels annotations.",
        "contributor": "l3i",
        "url": "https://ebdtheque.univ-lr.fr/",
        "date_created": now_formatted,
    }

    categories = [
        {
            "supercategory": None,
            "id": PANEL_ID,
            "name": "panel",
        },
        {
            "supercategory": None,
            "id": BUBBLE_ID,
            "name": "speech_bubble",
        },
        {
            "supercategory": None,
            "id": CHARACTER_ID,
            "name": "character",
        }
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
            args.ds_dir, f"annotations/{img_basename.split('.')[0]}.svg")

        if not os.path.exists(ann_path):
            continue

        with minidom.parse(ann_path) as ann_f:
            svg = ann_f.getElementsByTagName("svg")
            panels = next(filter(
                lambda x: x.getAttribute('class') == "Panel",
                svg
            ))
            characters = next(filter(
                lambda x: x.getAttribute('class') == "Character",
                svg
            ))
            balloons = next(filter(
                lambda x: x.getAttribute('class') == "Balloon",
                svg
            ))

            for object_id, object_list in [
                (PANEL_ID, panels),
                (CHARACTER_ID, characters),
                (BUBBLE_ID, balloons)]:
                for ann in object_list.getElementsByTagName("polygon"):
                    ann = ann.getAttribute("points").split()

                    if ann[0] == ann[-1]:
                        ann = ann[:-1]

                    coords = [coord.split(',') for coord in ann]
                    x, y = zip(*coords)
                    x = [int(e) for e in x]
                    y = [int(e) for e in y]
                    max_x = max(x); min_x = min(x)
                    max_y = max(y); min_y = min(y)
                    coords = [
                        min_x, min_y,
                        max_x, max_y
                    ]
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
                        "category_id": object_id,
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
