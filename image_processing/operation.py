import os
import cv2
from django.conf import settings
import numpy as np

def apply_smoothing(image_path, mask_size):
    # Read the image
    image = cv2.imread(image_path)
    image_r = cv2.resize(image, (500, 500))

    # Save the resized original image
    resized_image_name = 'resized_' + os.path.basename(image_path)
    resized_image_path = os.path.join(settings.MEDIA_ROOT, resized_image_name)
    cv2.imwrite(resized_image_path, image_r) 

    # Apply smoothing (Gaussian Blur)
    smoothed_image = cv2.GaussianBlur(image_r, ((mask_size * 6) + 1, (mask_size * 6) + 1), mask_size)

    # Construct path to save in the MEDIA_ROOT directory
    smoothed_image_name = 'smoothed_' + os.path.basename(image_path)
    smoothed_image_path = os.path.join(settings.MEDIA_ROOT, smoothed_image_name)
    
    # Save the resulting image in the media folder
    cv2.imwrite(smoothed_image_path, smoothed_image)
    res = []
    res.append(str(resized_image_name))
    res.append(str(smoothed_image_name))

    return res


def apply_box(image_path, mask_size):
    # Read the image
    image = cv2.imread(image_path)
    image_r = cv2.resize(image, (500, 500))

    # Save the resized original image
    resized_image_name = 'resized_box' + os.path.basename(image_path)
    resized_image_path = os.path.join(settings.MEDIA_ROOT, resized_image_name)
    cv2.imwrite(resized_image_path, image_r) 

    # Apply Box Blur
    mask = np.ones((mask_size, mask_size)) / mask_size ** 2
    blur_image = cv2.filter2D(image_r, -1, mask)

    # Construct path to save in the MEDIA_ROOT directory
    blur_image_name = 'box_' + os.path.basename(image_path)
    blur_image_path = os.path.join(settings.MEDIA_ROOT, blur_image_name)
    
    # Save the resulting image in the media folder
    cv2.imwrite(blur_image_path, blur_image)
    res = []
    res.append(str(resized_image_name))
    res.append(str(blur_image_name))

    return res


def apply_dog(image_path, mask_size1, mask_size2):
    # Read the image
    image = cv2.imread(image_path)
    image_r = cv2.resize(image, (500, 500))

    image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
    # Save the resized original image
    resized_image_name = 'resized_dog' + os.path.basename(image_path)
    resized_image_path = os.path.join(settings.MEDIA_ROOT, resized_image_name)
    cv2.imwrite(resized_image_path, image_r) 

    # Apply smoothing  (Gaussian Blur)
    smoothed_image1 = cv2.GaussianBlur(image_r_gray, ((mask_size1 * 6) + 1, (mask_size1 * 6) + 1), mask_size1)
    smoothed_image2 = cv2.GaussianBlur(image_r_gray, ((mask_size2 * 6) + 1, (mask_size2 * 6) + 1), mask_size2)
    # Calculate the Difference of Gaussian (DoG)
    dog_image = cv2.absdiff(smoothed_image1, smoothed_image2)

    # Construct path to save in the MEDIA_ROOT directory
    dog_image_name = 'dog_' + os.path.basename(image_path)
    dog_image_path = os.path.join(settings.MEDIA_ROOT, dog_image_name)
    
    # Save the resulting image in the media folder
    cv2.imwrite(dog_image_path, dog_image)
    res = []
    res.append(str(resized_image_name))
    res.append(str(dog_image_name))

    return res

def apply_canny(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_r = cv2.resize(image, (500, 500))

    image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    median = np.median(image_r_gray)
    lower_threshold = int(max(0, 0.3 * median))
    upper_threshold = int(min(255, 1.7 * median))
    # Save the resized original image
    resized_image_name = 'resized_canny' + os.path.basename(image_path)
    resized_image_path = os.path.join(settings.MEDIA_ROOT, resized_image_name)
    cv2.imwrite(resized_image_path, image_r) 

    # Apply Canny detector
    canny_im = cv2.Canny(image_r_gray, lower_threshold, upper_threshold)

    # Construct path to save in the MEDIA_ROOT directory
    canny_image_name = 'dog_' + os.path.basename(image_path)
    canny_image_path = os.path.join(settings.MEDIA_ROOT, canny_image_name)
    
    # Save the resulting image in the media folder
    cv2.imwrite(canny_image_path, canny_im)
    res = []
    res.append(str(resized_image_name))
    res.append(str(canny_image_name))

    return res


def apply_stitching(image_paths):
    # Read images from file paths
    images = [cv2.imread(image_path) for image_path in image_paths]

    # Stitch the images using OpenCV's Stitcher
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        # Map the status code to the possible issues
        error_message = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images to stitch",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
        }.get(status, "Unknown stitching error")
        
        raise Exception(f"Error stitching images: {error_message}")

    # Crop the stitched image to remove unwanted curved edges
    stitched_image_cropped = crop_image(stitched_image)

    # Save the stitched image
    stitched_image_path = os.path.join(os.path.dirname(image_paths[0]), 'stitched_result.jpg')
    cv2.imwrite(stitched_image_path, stitched_image_cropped)

    return image_paths, stitched_image_path

def crop_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define margins to crop (adjust these values as needed)
    top_margin = int(height * 0.04)  # 4% from the top
    bottom_margin = int(height * 0.04)  # 4% from the bottom
    left_margin = int(width * 0.015)  # 1% from the left
    right_margin = int(width * 0.015)  # 1% from the right

    # Crop the image
    cropped_image = image[top_margin:height - bottom_margin, left_margin:width - right_margin]

    return cropped_image







def apply_morphology(image, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_GRADIENT, (kernel_size, kernel_size))
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morphed






