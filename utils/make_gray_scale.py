import os

from PIL import Image


def convert_images_to_grayscale(directory: str) -> None:
    """
    Convert all images in a directory to grayscale.

    Parameters
    ----------
        directory : str
            The directory containing images to convert.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Open the image
            image = Image.open(file_path)
            # Check if the image is already grayscale
            if image.mode != "L":
                # Convert to grayscale
                gray_image = image.convert("L")
                # Save the grayscale image, replacing the original file
                gray_image.save(file_path)
                print(f"Converted {filename} to grayscale.")
            else:
                print(f"{filename} is already grayscale.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert images in a directory to grayscale."
    )
    parser.add_argument(
        "directory", type=str, help="The directory containing images to convert."
    )

    args = parser.parse_args()

    convert_images_to_grayscale(args.directory)
