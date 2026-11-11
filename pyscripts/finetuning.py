from os import path

from ultralytics import YOLO

from lib import env, utils


def finetune(name, weights_path, yaml_path, freeze):
    model = YOLO(weights_path)
    model.train(
        data=yaml_path,
        epochs=env.EPOCHS,
        patience=env.PATIENCE,
        fraction=env.FRACTION,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name=name,
        amp=True,
        batch=-1,
        cache=env.CACHE,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.001,
    )
    return model


def fintune_and_evaluate(freeze):
    test_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_TEST_PROJECT,
        version_n=env.TEST_PROJECT_VERSION,
    )

    w_path = "pastData/sizes/training2-for-imgsz-1280/training-online-dataset-2/weights/best.pt"
    model = finetune(
        name=f"finetune-mytask-freeze-{freeze}",
        weights_path=w_path,
        yaml_path=path.join(test_dataset.location, "data.yaml"),
        freeze=freeze,
    )

    utils.evaluate(
        name=f"val-finetuned-freeze-{freeze}",
        model=model,
        dataset_dir=test_dataset.location,
        split="val",
    )
    utils.show_predictions(
        name=f"show-finetuned-preds-{freeze}",
        model=model,
        imgs_dir=path.join(test_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )


def main():
    freezes = [23]
    for freeze in freezes:
        fintune_and_evaluate(freeze)


if __name__ == "__main__":
    main()
