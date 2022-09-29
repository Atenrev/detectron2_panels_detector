import fiftyone as fo
import fiftyone.utils.coco as fouc


seed = 42

# The directory containing the dataset to import
dataset_dir = "./datasets/eBDtheque_database_v3"

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
