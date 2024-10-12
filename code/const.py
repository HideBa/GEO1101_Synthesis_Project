import os

program_dir = os.getcwd()

# Cache directory will be set by the main script based on the training set
cache_dir = os.path.join(program_dir, "data", "cache")

# Ensure the cache directory exists (done in the main script)
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Updated cache paths based on the cache directory
VOCABULARY_CACHE_PATH = os.path.join(
    cache_dir, "vocabulary.pickle"
)
DATABASE_CACHE_PATH = os.path.join(
    cache_dir, "database.pickle"
)
IMAGE_NAMES_CACHE_PATH = os.path.join(
    cache_dir, "image_name_index_pairs.npy"
)

# Paths that are unrelated to the cache remain the same
GROUND_TRUTH_PATH = os.path.join(
    program_dir, "data", "csvs", "slam_camera_coordinates.csv"
)

WALL_COORDINATES_PATH = os.path.join(
    program_dir, "data", "csvs", "BK_wall_coordinates.csv"
)

VALIDATION_FILE_PATH = os.path.join(
    program_dir, "data", "csvs", "manual_validation.csv"
)