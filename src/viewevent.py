"""
MiniPIX EDU Event Viewer
------------------------

Step 1 of the Cosmic Muon 3D Reconstruction project.

This program:
1. Loads a PIXet-exported ASCII/CSV event.
2. Converts the pixel data into a 256 x 256 detector image.
3. Displays the particle track.
4. Calculates some basic properties.
5. Saves the processed event as a NumPy file.

The MiniPIX EDU uses a 256 x 256 Timepix detector.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

IMAGE_SIZE = 256

# Change this to the file exported from PIXet.
INPUT_FILE = Path("data/raw/event_0001.txt")

# Where processed events will be saved.
OUTPUT_FILE = Path("data/processed/event_0001.npy")


# ---------------------------------------------------------
# LOAD PIXEL DATA
# ---------------------------------------------------------

def load_event(filename):
    """
    Load an event exported from PIXet.

    Expected basic format:

        x   y   value

    where:
        x = pixel x-coordinate
        y = pixel y-coordinate
        value = measured pixel value

    If your PIXet export uses a different format,
    we'll modify this function after looking at one
    of your actual files.
    """

    print(f"Loading: {filename}")

    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            "The file needs at least three columns: x, y, value"
        )

    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    value = data[:, 2].astype(float)

    return x, y, value


# ---------------------------------------------------------
# CREATE IMAGE
# ---------------------------------------------------------

def make_image(x, y, value):
    """
    Convert individual pixel measurements into
    a 256 x 256 image.
    """

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    valid = (
        (x >= 0) &
        (x < IMAGE_SIZE) &
        (y >= 0) &
        (y < IMAGE_SIZE)
    )

    x = x[valid]
    y = y[valid]
    value = value[valid]

    # Multiple measurements can theoretically fall
    # into the same pixel, so accumulate them.
    np.add.at(image, (y, x), value)

    return image


# ---------------------------------------------------------
# BASIC EVENT ANALYSIS
# ---------------------------------------------------------

def analyze_event(image):
    """
    Calculate simple properties of the event.
    """

    total_signal = np.sum(image)

    active_pixels = np.count_nonzero(image)

    if total_signal == 0:
        return {
            "total_signal": 0,
            "active_pixels": 0,
            "center_x": None,
            "center_y": None,
        }

    y_indices, x_indices = np.indices(image.shape)

    center_x = np.sum(x_indices * image) / total_signal
    center_y = np.sum(y_indices * image) / total_signal

    return {
        "total_signal": total_signal,
        "active_pixels": active_pixels,
        "center_x": center_x,
        "center_y": center_y,
    }


# ---------------------------------------------------------
# DISPLAY EVENT
# ---------------------------------------------------------

def display_event(image, properties):
    """
    Display the MiniPIX event.
    """

    plt.figure(figsize=(8, 8))

    plt.imshow(
        image,
        origin="lower",
        interpolation="nearest"
    )

    plt.title("MiniPIX EDU Particle Event")

    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")

    plt.colorbar(label="Pixel signal")

    # Show center of deposited signal.
    if properties["center_x"] is not None:
        plt.scatter(
            properties["center_x"],
            properties["center_y"],
            marker="+",
            s=100
        )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        print()
        print("ERROR: Event file not found.")
        print()
        print(f"Expected file:")
        print(f"    {INPUT_FILE}")
        print()
        print("Export an event from PIXet and put it there.")
        return

    # Load raw event
    x, y, value = load_event(INPUT_FILE)

    # Convert to detector image
    image = make_image(x, y, value)

    # Analyze
    properties = analyze_event(image)

    print()
    print("========== EVENT ==========")
    print(f"Active pixels : {properties['active_pixels']}")
    print(f"Total signal  : {properties['total_signal']:.2f}")

    if properties["center_x"] is not None:
        print(f"Center X      : {properties['center_x']:.2f}")
        print(f"Center Y      : {properties['center_y']:.2f}")

    # Save processed event
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(OUTPUT_FILE, image)

    print()
    print(f"Processed event saved to:")
    print(f"    {OUTPUT_FILE}")

    # Display
    display_event(image, properties)


if __name__ == "__main__":
    main()
