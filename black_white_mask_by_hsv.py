import cv2
import numpy as np
import os

def convert_red_mask_to_binary(input_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read input image
    img = cv2.imread(input_path)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    binary_mask = cv2.bitwise_or(mask1, mask2)

    # Optional: ensure mask is strictly 0 or 255
    binary_mask[binary_mask > 0] = 255

    # Get output path
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    # Save binary mask
    cv2.imwrite(output_path, binary_mask)

    print(f"Saved binary mask to: {output_path}")

convert_red_mask_to_binary('PlantVillage_masks/E_B/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.jpg', 'C:/Users/Lenovo/OneDrive/Desktop/Project/output_test')
