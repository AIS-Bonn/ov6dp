import matplotlib
import torch
import cv2
import numpy as np

import pykinect_azure as pykinect

import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image capture and save with OpenCV.")
    parser.add_argument("--prefix", type=str, default="image", help="Prefix for saved image filenames.")
    args = parser.parse_args()
    
    prefix = args.prefix
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    image_count = 0  # Counter for saved images

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)
    
    print("Press 's' to save the current image.")
    print("Press 'q' to quit.")

    while True:
        # Acquire an image
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not (ret_color and ret_depth):
            continue
        
        # Display the image
        cv2.imshow("Color Image", color_image)

        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF  # Mask unnecessary bits

        if key == ord('s'):  # Save image if 's' is pressed
            image_path = os.path.join(save_dir, f"{prefix}_depth_{image_count}.npy")
            np.save(image_path, transformed_depth_image)

            image_path = os.path.join(save_dir, f"{prefix}_rgb_{image_count}.jpg")
            cv2.imwrite(image_path, color_image)
            
            print(f"Image saved: {image_path}")
            image_count += 1
        elif key == ord('q'):  # Quit if 'q' is pressed
            print("Exiting program.")
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
