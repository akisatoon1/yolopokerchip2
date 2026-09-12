from os.path import join

from ultralytics import YOLO

from lib import utils
from lib.env import ROBOFLOW_COLOR_PROJECT, COLOR_PROJECT_VERSION


def main():
    dataset = utils.download_dataset(
        project_name=ROBOFLOW_COLOR_PROJECT,
        version_n=COLOR_PROJECT_VERSION,
    )
    model = utils.train_model(
        name="color-trained",
        dataset_dir=dataset.location,
    )

    best_model = YOLO(model.trainer.best)

    utils.evaluate(
        name="val-color",
        model=best_model,
        dataset_dir=dataset.location,
        split="val",
    )
    utils.show_predictions(
        name="show-color-preds",
        model=best_model,
        imgs_dir=join(dataset.location, "valid", "images"),
    )


if __name__ == "__main__":
    main()
