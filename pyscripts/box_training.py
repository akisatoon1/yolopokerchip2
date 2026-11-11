from os.path import join

from ultralytics import YOLO

from lib import env, utils


def main():
    dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_BOX_PROJECT,
        version_n=env.BOX_PROJECT_VERSION,
    )
    model = utils.train_model(
        name="training-box",
        dataset_dir=dataset.location,
    )
    model_best = YOLO(model.trainer.best)
    utils.show_predictions(
        name="show-box",
        model=model_best,
        imgs_dir=join(dataset.location, "test", "images"),
        box_or_seg="box",
    )


if __name__ == "__main__":
    main()
