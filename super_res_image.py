#!/usr/bin/env python

# orig: https://pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
# more straightforward: https://learnopencv.com/super-resolution-in-opencv/

import argparse
from functools import lru_cache
import time
import cv2
import os


SuperResModel = cv2.dnn_superres.DnnSuperResImpl


@lru_cache
def load_model(model_path: str) -> SuperResModel:
    file_name = os.path.basename(model_path)
    model_name = file_name.split("_")[0].lower()
    model_scale = int(file_name.split("_x")[-1].removesuffix(".pb"))
    print(f"[INFO] loading {model_name=} {model_scale=}")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, model_scale)
    return sr


def bicubic_resize(image: cv2.Mat, width: int, height: int) -> cv2.Mat:
    # resize the image using standard bicubic interpolation
    start = time.time()
    bicubic = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    print(f"[INFO] bicubic interpolation took {time.time() - start:.3f} seconds")
    return bicubic


def save_upsampled(image_path: str, model: SuperResModel) -> None:
    image = cv2.imread(image_path)
    print("[INFO] resizing orig dimensions:", image.shape)
    start = time.time()
    upscaled = model.upsample(image)
    print(f"[INFO] upscaled: {upscaled.shape}; took {time.time()-start:.6f} seconds")
    orig_image_base = os.path.splitext(image_path)[0]
    upscaled_path = (
        orig_image_base + f"_{model.getAlgorithm()}x{model.getScale()}"
        f"_{upscaled.shape[1]}x{upscaled.shape[0]}.jpg"
    )
    print("[INFO] saving to:", upscaled_path)
    cv2.imwrite(upscaled_path, upscaled)


def compare_4x(image_path: str) -> None:
    paths_4x = [f"models/{p}" for p in os.listdir("models") if "_x4" in p]
    for model_path_4x in paths_4x:
        save_upsampled(image_path, load_model(model_path_4x))


# construct the argument parser and parse the arguments
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model", required=True, help="path to super resolution model"
    )
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="path to input image we want to increase resolution of",
    )
    args = ap.parse_args()

    sr = load_model(args.model)
    save_upsampled(args.image, sr)


if __name__ == "__main__":
    main()
