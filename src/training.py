import os

from ultralytics import YOLO

from lib import env, utils


def main():
    pretrained_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_PRETRAINED_PROJECT,
        version_n=env.PRETRAINED_PROJECT_VERSION,
    )
    test_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_TEST_PROJECT,
        version_n=env.TEST_PROJECT_VERSION,
    )

    model = utils.train_model(
        name="pretrained",
        dataset_dir=pretrained_dataset.location,
    )
    model_best = YOLO(model.trainer.best)

    if env.DEVICE == "cuda":
        # cpuのときはなぜかここでKilledされる
        utils.evaluate(
            name="val-pretrained",
            model=model_best,
            dataset_dir=pretrained_dataset.location,
            split="val",
        )
    utils.evaluate(
        name="val-mytask",
        model=model_best,
        dataset_dir=test_dataset.location,
        split="val",
    )
    utils.show_predictions(
        name="show-val-predictions",
        model=model_best,
        imgs_dir=os.path.join(pretrained_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )
    utils.show_predictions(
        name="show-mytask-predictions",
        model=model_best,
        imgs_dir=os.path.join(test_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )


if __name__ == "__main__":
    main()
