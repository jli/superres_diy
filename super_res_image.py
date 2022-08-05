#!/usr/bin/env python

# orig: https://pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
# more straightforward: https://learnopencv.com/super-resolution-in-opencv/

import argparse
import os
import time
from functools import lru_cache
from typing import Literal

import cv2


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


def compare_models(image_path: str, scale: Literal[2, 4]) -> None:
    model_paths = [f"models/{p}" for p in os.listdir("models") if f"_x{scale}" in p]
    for model_path in model_paths:
        save_upsampled(image_path, load_model(model_path))


# https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv
def unsharp_mask(
    image: cv2.Mat, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0
):
    import numpy as np

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred  # pyright: ignore
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


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
