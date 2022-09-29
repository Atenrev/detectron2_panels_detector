import argparse
import json
import os
import numpy as np
from math import ceil

from tqdm import tqdm
from shutil import copyfile


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for splitting COCO datasets.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--ds_dir", "-m", type=str, default="./datasets/eBDtheque_database_v3_99",
                        help="Dataset dir")
    parser.add_argument("--train_split", "-ts", type=float, default=0.7,
                        help="Dataset dir")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Dataset dir")

    return parser.parse_args()


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    with open(os.path.join(args.ds_dir, "labels.json"), "r") as f:
        labels_org = json.load(f)

    labels_train = dict(labels_org)
    labels_test = dict(labels_org)

    labels_train["images"] = []
    labels_test["images"] = []

    new_indices = np.random.permutation(len(labels_org["images"]))
    split_train = ceil(len(labels_org["images"]) * args.train_split)
    labels_org_arr = np.array(labels_org["images"])[new_indices]
    labels_train["images"] = labels_org_arr[:split_train].tolist()
    train_image_ids = [im["id"] for im in labels_train["images"]]
    labels_test["images"] = labels_org_arr[split_train:].tolist()
    test_image_ids = [im["id"] for im in labels_test["images"]]

    labels_train["annotations"] = [
        ann for ann in labels_org["annotations"]
        if ann["image_id"] in train_image_ids
    ]
    labels_test["annotations"] = [
        ann for ann in labels_org["annotations"]
        if ann["image_id"] in test_image_ids
    ]
    
    images_org_path = os.path.join(args.ds_dir, "data")
    train_path = f"{args.ds_dir}_train"
    images_train_path = os.path.join(train_path, "data")
    test_path = f"{args.ds_dir}_test"
    images_test_path = os.path.join(test_path, "data")
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_test_path, exist_ok=True)

    for im in tqdm(labels_train["images"]):
        source = os.path.join(images_org_path, im["file_name"])
        dest = os.path.join(images_train_path, im["file_name"])
        copyfile(source, dest)

    for im in tqdm(labels_test["images"]):
        source = os.path.join(images_org_path, im["file_name"])
        dest = os.path.join(images_test_path, im["file_name"])
        copyfile(source, dest)

    with open(os.path.join(train_path, "labels.json"), "w") as f:
        json.dump(labels_train, f)

    with open(os.path.join(test_path, "labels.json"), "w") as f:
        json.dump(labels_test, f)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
