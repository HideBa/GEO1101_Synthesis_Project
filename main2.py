import cv2
import numpy as np
import os

# Define input and output folders
input_folder = r'D:\geomatics\geo1101\panoramic\panoramas'  # Replace with your input folder path
output_folder = r'D:\geomatics\geo1101\panoramic\panoramas_output'  # Replace with your output folder path

# Function to project equirectangular image to cube map face using NumPy
def equirectangular_to_cubemap(image, face_size, image_name, output_folder):
    h, w = image.shape[:2]

    # Define cube face names
    faces = ['front', 'right', 'rear', 'left', 'top', 'bottom']

    # Empty list to store cube faces
    cubemap_faces = []

    # For each face, calculate the view direction and project it
    for face in faces:
        # Create an empty array for the face
        cubemap_face = np.zeros((face_size, face_size, 3), dtype=np.uint8)

        # Create a meshgrid for pixel coordinates
        i, j = np.meshgrid(np.arange(face_size), np.arange(face_size), indexing='ij')

        # Convert pixel coordinates (i, j) to 3D direction vectors using NumPy
        x, y, z = pixel_to_direction(face, i, j, face_size)

        # Convert 3D directions to 2D equirectangular coordinates
        u, v = direction_to_equirectangular(x, y, z, h, w)

        # Clamp u and v to valid image indices
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)

        # Use the computed coordinates to sample the panorama image
        cubemap_face[:, :] = image[v.astype(int), u.astype(int)]

        # Orientation correction
        if face in ['top', 'bottom']:
            cubemap_face = np.fliplr(cubemap_face)  # Flip horizontally

        if face in ['front', 'rear', 'right', 'left','top', 'bottom']:
            cubemap_face = cv2.rotate(cubemap_face, cv2.ROTATE_180)  # Flip horizontally

        # Rotate each face by 90 degrees counter clockwise
        cubemap_face = cv2.rotate(cubemap_face, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Append the face to the list
        cubemap_faces.append(cubemap_face)

        # Save each face directly to the output folder with a unique name
        output_filename = os.path.join(output_folder, f'{image_name}_{face}.jpg')
        cv2.imwrite(output_filename, cubemap_face)

    return cubemap_faces

# Function to convert pixel to 3D direction vector for a specific face
def pixel_to_direction(face, i, j, face_size):
    a = 2 * (i / face_size) - 1
    b = 2 * (j / face_size) - 1

    if face == 'front':
        x, y, z = -a, -b, np.ones_like(a)
    elif face == 'right':
        x, y, z = np.ones_like(a), -b, a
    elif face == 'rear':
        x, y, z = a, -b, -np.ones_like(a)
    elif face == 'left':
        x, y, z = -np.ones_like(a), -b, -a
    elif face == 'top':
        x, y, z = a, np.ones_like(a), b
    elif face == 'bottom':
        x, y, z = a, -np.ones_like(a), -b

    # Normalize the vectors using NumPy's vectorized operations
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x / norm, y / norm, z / norm

# Function to map 3D direction vector to 2D equirectangular coordinates
def direction_to_equirectangular(x, y, z, height, width):
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    u = (lon / (2 * np.pi) + 0.5) * width
    v = (0.5 - lat / np.pi) * height

    return u, v

# Function to process all images in the input folder
def process_images_from_folder(input_folder, output_folder, face_size=960):
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a valid image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.splitext(filename)[0]  # Get image name without extension

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue

            # Process the image to generate cubemap faces
            print(f"Processing {filename}...")
            equirectangular_to_cubemap(image, face_size, image_name, output_folder)

# Run the processing for images in the folder
process_images_from_folder(input_folder, output_folder)
