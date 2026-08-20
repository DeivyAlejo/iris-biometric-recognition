from pathlib import Path

import cv2 as cv
import numpy as np


DATASET_DIRECTORY = Path("dataset")
OUTPUT_DIRECTORY = Path("iris_detection_results")

# Change this value to use another blur configuration.
SELECTED_BLUR = "35x35"

# These thresholds match param1=50 in HoughCircles.
CANNY_LOW_THRESHOLD = 25
CANNY_HIGH_THRESHOLD = 50

BLUR_OPTIONS = {
    "21x21": ((21, 21), 3),
    "27x27": ((27, 27), 4),
    "35x35": ((35, 35), 5),
    "43x43": ((43, 43), 7),
    "51x51": ((51, 51), 8),
}


def normalize_red_channel(image):
    """Extract the red channel and stretch its contrast."""
    red_channel = image[:, :, 2]
    low, high = np.percentile(red_channel, (1, 99))

    if high <= low:
        return red_channel.copy()

    normalized = np.clip(
        (red_channel.astype(np.float32) - low) / (high - low),
        0,
        1,
    )
    return (normalized * 255).astype(np.uint8)


def blur_image(image, blur_name=SELECTED_BLUR):
    """Normalize the red channel and apply the selected Gaussian blur."""
    if blur_name not in BLUR_OPTIONS:
        available_options = ", ".join(BLUR_OPTIONS)
        raise ValueError(
            f"Unknown blur option {blur_name!r}. Choose one of: "
            f"{available_options}"
        )

    layer = normalize_red_channel(image)
    kernel_size, sigma = BLUR_OPTIONS[blur_name]
    return cv.GaussianBlur(layer, kernel_size, sigma)


def detect_canny_edges(image, blur_name=SELECTED_BLUR):
    """Return the blurred image and the edges used to inspect detection."""
    blurred = blur_image(image, blur_name)
    edges = cv.Canny(
        blurred,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )
    return blurred, edges


def detect_iris(image, blur_name=SELECTED_BLUR):
    """Detect and return Hough's strongest iris circle as (x, y, radius)."""
    blurred = blur_image(image, blur_name)
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=CANNY_HIGH_THRESHOLD,
        param2=30,
        minRadius=90,
        maxRadius=150,
    )

    if circles is None:
        return None

    # HoughCircles orders candidates by accumulator strength. The first
    # candidate was the most reliable one during the dataset evaluation.
    x, y, radius = np.rint(circles[0][0]).astype(int)
    return int(x), int(y), int(radius)


def draw_iris_circle(image, circle):
    """Return a copy of the image with the iris and its center marked."""
    result = image.copy()

    if circle is not None:
        x, y, radius = circle
        cv.circle(result, (x, y), radius, (0, 255, 0), 2)
        cv.circle(result, (x, y), 3, (0, 0, 255), -1)

    return result


def crop_iris_circle(image, circle):
    """Crop the iris bounding square and mask pixels outside its circle."""
    if circle is None:
        return None

    x, y, radius = circle
    image_height, image_width = image.shape[:2]

    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = min(x + radius + 1, image_width)
    y2 = min(y + radius + 1, image_height)

    cropped = image[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        return None

    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    center_in_crop = (x - x1, y - y1)
    cv.circle(mask, center_in_crop, radius, 255, -1)

    return cv.bitwise_and(cropped, cropped, mask=mask)


def process_image(image_path, output_path, blur_name=SELECTED_BLUR):
    """Detect the iris in one image and save its circular crop."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return None

    circle = detect_iris(image, blur_name)
    result = crop_iris_circle(image, circle)

    if result is None:
        result = image.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), result):
        raise OSError(f"Could not save result to {output_path}")

    if circle is None:
        print(f"No iris detected: {image_path}")
    else:
        x, y, radius = circle
        print(
            f"Detected {image_path}: center=({x}, {y}), "
            f"radius={radius}"
        )

    return circle


def image_sort_key(image_path):
    """Sort numeric subject and image directory names numerically."""
    try:
        return int(image_path.parent.name), int(image_path.stem)
    except ValueError:
        return image_path.parent.name, image_path.stem


def process_dataset(
    dataset_directory=DATASET_DIRECTORY,
    output_directory=OUTPUT_DIRECTORY,
    blur_name=SELECTED_BLUR,
):
    """Detect irises in every TIFF image without overwriting the originals."""
    dataset_directory = Path(dataset_directory)
    output_directory = Path(output_directory)
    image_paths = sorted(
        dataset_directory.glob("*/*.tiff"),
        key=image_sort_key,
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No TIFF images found under {dataset_directory}"
        )

    detected_count = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(dataset_directory)
        output_path = (output_directory / relative_path).with_suffix(".jpg")
        circle = process_image(image_path, output_path, blur_name)
        detected_count += circle is not None

    print(
        f"Finished: detected {detected_count}/{len(image_paths)} irises. "
        f"Results saved under {output_directory}"
    )
    return detected_count, len(image_paths)


def main():
    process_dataset()


if __name__ == "__main__":
    main()
