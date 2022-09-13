import json
import fiftyone as fo
import fiftyone.utils.coco as fouc


# The directory containing the dataset to import
dataset_dir = "./datasets/comics_panels_annotations"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    # label_types="segmentations"
)

# Add COCO predictions to `predictions` field of dataset
classes = dataset.default_classes
fouc.add_coco_labels(
    dataset,
    "predictions",
    "output/inference/coco_instances_results.json",
    classes,
    # coco_id_field="id"
)

session = fo.launch_app(dataset)
session.wait()
