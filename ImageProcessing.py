import cv2
import numpy as np


def gradient_blur_upper_half(image_path, blur_strength=(21, 21), save_path=None):
    """
    Apply a gradient blur to the upper half of an image.
    
    Args:
        image_path (str): Path to the input image
        blur_strength (tuple): Kernel size for Gaussian blur (width, height)
        save_path (str, optional): Path to save the output image. If None, image is displayed.

    Returns:
        The processed image with upper half gradient blur
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create a blurred version of the entire image
    blurred = cv2.GaussianBlur(img, blur_strength, 0)

    # Get image dimensions
    height, width = img.shape[:2]
    half_height = height // 2

    # Create a gradient mask (white at top, black at middle)
    mask = np.zeros((height, width), dtype=np.float32)

    # Fill the upper half with a gradient (1.0 at top, 0.0 at middle)
    for y in range(half_height):
        # Calculate gradient value (1.0 at top, 0.0 at middle)
        gradient_value = 1.0 - (y / half_height)
        mask[y, :] = gradient_value

    # Expand mask to 3 channels if the image has 3 channels
    if len(img.shape) == 3:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)

    # Blend the original and blurred images using the mask
    # Formula: result = mask * blurred + (1 - mask) * original
    result = (mask * blurred + (1 - mask) * img).astype(np.uint8)

    # Save or display the result
    if save_path:
        cv2.imwrite(save_path, result)
        print(f"Saved result to {save_path}")
    else:
        # Display the result
        cv2.imshow("Original", img)
        cv2.imshow("Gradient Blur", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "your_image.jpg"
    # Increase blur_strength for more intense blur
    gradient_blur_upper_half(image_path, blur_strength=(31, 31))