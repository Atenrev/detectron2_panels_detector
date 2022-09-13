import os

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from src.augmentations import MyColorAugmentation


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    return COCOEvaluator(dataset_name, output_dir=output_folder)


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[ 
            MyColorAugmentation(),
            T.RandomBrightness(0.125, 8),
            T.RandomContrast(0.125, 8),
            T.RandomSaturation(0.125, 8),
        ])
        return build_detection_train_loader(cfg, mapper=mapper)