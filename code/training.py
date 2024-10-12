import glob
import os
import time

import cv2
import dbow
import numpy as np
from const import (
    DATABASE_CACHE_PATH,
    IMAGE_NAMES_CACHE_PATH,
    VOCABULARY_CACHE_PATH,
)
from tqdm import tqdm


def training(
        dataset_dir: str,
        cache_dir: str,
        cluster_num: int = 10,
        depth=2,
) -> None:
    print(f"===Loading Images from {dataset_dir}===")
    program_dir = os.getcwd()
    orb = cv2.ORB_create()
    dataset_dir = os.path.join(program_dir, dataset_dir)
    png_path, jpeg_path = os.path.join(dataset_dir, "*.png"), os.path.join(
        dataset_dir, "*.jpg"
    )
    image_paths = glob.glob(png_path) + glob.glob(jpeg_path)
    images = []
    image_names = np.array([])
    for image_path in tqdm(image_paths, desc="adding image to images"):
        images.append(cv2.imread(image_path))
        image_name = os.path.basename(image_path)
        image_names = np.append(image_names, image_name)

    vocabulary = dbow.Vocabulary(images, cluster_num, depth)
    print("===Vocabulary has been made===")
    print("===Database is being created===")

    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Use dynamic cache paths
    vocab_path = os.path.join(cache_dir, "vocabulary.pickle")
    image_name_path = os.path.join(cache_dir, "image_name_index_pairs.npy")
    database_path = os.path.join(cache_dir, "database.pickle")

    vocabulary.save(vocab_path)
    np.save(image_name_path, image_names)

    db = dbow.Database(vocabulary)
    invalid_images = []
    for i, image in tqdm(enumerate(images), desc="Processing images"):
        _, descs = orb.detectAndCompute(image, None)
        if descs is None:
            invalid_images.append(image_paths[i])
            continue
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        db.add(descs)
    db.save(database_path)


def database(database_path: str) -> dbow.Database:
    if os.path.exists(database_path):
        return dbow.Database.load(database_path)
    else:
        training()
        return dbow.Database.load(database_path)


if __name__ == "__main__":
    start_time = time.time()  # Start timing
    print("training started: ", start_time)

    # Train with different datasets and cache folders
    training_sets = ["left_only", "right_only", "front_only", "top_only"]
    for training_set in training_sets:
        dataset_dir = os.path.join("data", training_set, "images")
        cache_dir = os.path.join("data", training_set, "cache")
        training(dataset_dir, cache_dir)

    # Calculate and format elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    )  # Print in hh:mm:ss
