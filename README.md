# AI-Enhanced Image Stitching and Edge Detection Project

## Table of Contents

1. [Introduction](#1-introduction)
2. [Set Up](#2-setup)
3. [Technologies Used](#3-technologies-used)
4. [Features Implemented](#4-features-implemented)
   - [3.1 Image Stitching](#41-image-stitching)
   - [3.2 Edge Detection](#42-edge-detection)
     - [Canny Edge Detection](#canny-edge-detection)
     - [DoG with Morphological Operations](#dog-with-morphological-operations)
   - [3.3 AI-Based Human Detection](#43-ai-based-human-detection)
5. [User Interface & Workflow](#5-user-interface--workflow)
6. [Challenges and Solutions](#6-challenges-and-solutions)
   - [1. Path Handling Issues](#1-path-handling-issues)
   - [2. Edge Detection Complexity](#2-edge-detection-complexity)
   - [3. Human Detection Model Integration](#3-human-detection-model-integration)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

This project is designed to perform advanced image processing tasks using Django and OpenCV. It aims to provide an intuitive user interface for stitching images, applying edge detection techniques, and conducting AI-based human figure detection within stitched images. The project leverages traditional edge detection methods like Canny and Difference of Gaussians (DoG), as well as an AI model for detecting human figures, ensuring flexibility and robustness in image analysis.

---
## 2. Set Up:
- After downloading the folder provided, in the same directory where you saved the folder create virtual enviroment, so that you can download django, numpy, matplotlib and yolov8.
- now navigate to cv_project folder in the termial using cd cv_project, then to activate the server run py manage.py runserver, then on your browser wirte localhost:8000 and that should get you the full website.

## 3. Technologies Used

- **Django**: Web framework used to handle the back-end logic and provide a user interface for interaction.
- **OpenCV**: Open-source library used for image processing tasks, including edge detection and AI-based human detection.
- **YOLOv3**: Pre-trained object detection model used to detect human figures in the stitched image.
- **Python**: Programming language used to implement all backend logic.
- **HTML/CSS**: Frontend design for user interaction and presentation.

---

## 3. Features Implemented

### 3.1 Image Stitching

This step allows users to upload multiple images, which are then stitched together into a single, panoramic image. The stitching operation combines images with overlapping regions using OpenCV's `Stitcher` class.

**Implementation Overview:**
- Users upload images via a form.
- The images are passed to OpenCV’s `Stitcher` object, and a stitched result is generated.
- The stitched result is displayed alongside the original images for comparison.

---

### 3.2 Edge Detection

Two primary edge detection methods are applied to the stitched image:

- **Canny Edge Detection**
- **Difference of Gaussians (DoG)**

The results are displayed in a side-by-side format, allowing users to compare the results of each method.

### Canny Edge Detection

- This method identifies edges by looking for areas of rapid intensity change.
- The user uploads an image, and OpenCV’s `Canny()` function is applied to detect edges.

### DoG with Morphological Operations

- A Difference of Gaussians approach is followed by morphological operations like erosion and dilation.
- A slider is provided to allow users to dynamically adjust the kernel size of the morphological operation, refining the edge detection process.

**Morphology Slider UI:**

- A slider is created using OpenCV's `createTrackbar()`, allowing real-time kernel size adjustment.
- Users can modify the kernel size and instantly see the updated results on the processed image.


### 3.3 AI-Based Human Detection

YOLOv3 is used to detect human figures within the stitched image, with a confidence level threshold of 50% to ensure accurate results. The detected human figures are marked on the image with bounding boxes.

### Steps Involved:

1. Load YOLOv3 model and the corresponding `.cfg` and `.weights` files.
2. Pass the stitched image through the model.
3. Filter out detections with confidence levels below 50%.
4. Draw bounding boxes around detected human figures.


# 4. User Interface & Workflow

## Home Page
![im5](https://github.com/user-attachments/assets/1e3be30c-9644-41a6-901e-023072ae3120)



The home page offers multiple image processing operations, allowing users to select a task and upload images for processing. Each operation redirects the user to its corresponding page where they can upload an image, configure the parameters, and view the results.

### Available Operations

- **Box Blur**  
  Users can click on the "Box Blur" link, which directs them to a page where they can upload an image and select a mask size to apply the box blur effect.

- **Canny Edge Detection**  
  A link takes users to a page where they can upload an image and apply Canny Edge Detection, highlighting areas of rapid intensity change.

- **Difference of Gaussians (DoG)**  
  Users can upload an image and apply the Difference of Gaussians (DoG) method. They are provided with a slider to adjust the kernel size for the subsequent morphological operation (erosion/dilation) to refine the edge detection.

- **Image Stitching**  
  Users can upload multiple images, which are then stitched together to create a single panorama-like image. 

- **Full Workflow: Stitching, Canny, DoG & Morphological Operations**  
  This combined operation starts by stitching images together, followed by applying Canny Edge Detection, DoG, and morphological operations. A slider allows users to dynamically adjust the kernel size for the morphological operation to modify the final result.

- **Human Detection (Latest Stitched Image)**  
  This operation runs human detection on the latest stitched image using an AI-based object detection model (YOLOv3). The system filters out any detections with less than 50% confidence and displays bounding boxes around detected human figures.

- **Human Detection (Upload New Image)**  
  Users can upload a new image, and the human detection model (YOLOv3) will run on the image. Only detections with confidence levels above 50% are displayed with bounding boxes marking human figures.

  ## User Authentication and Profile Management

Our application supports user authentication, allowing users to register, log in, and log out. Each user has a personal profile that they can update at any time. Below are the features available:

1. **User Registration**: New users can create an account by providing their details. 
   ![im1](https://github.com/user-attachments/assets/d11c6d5c-1b05-4735-a254-08c3bdfdbb48)
![im2](https://github.com/user-attachments/assets/ae4e95bb-caea-433f-9873-4e077ed08f66)

2. **Login**: Registered users can log in using their credentials.
![im3](https://github.com/user-attachments/assets/7747ba94-c673-414c-b0b8-98e3c00cefc9)

3. **Logout**: Users can securely log out of their account at any time.

4. **Profile Management**: Each user has a profile that can be updated with their information. Users can edit their profile details as needed.
![im4](https://github.com/user-attachments/assets/4de10ae8-db3b-420f-89e6-95273360143f)

This functionality ensures a personalized experience for users while maintaining security and privacy.

# 5. Challenges and Solutions

### 1. Path Handling Issues

- **Issue**: File paths were being passed with their entire system-specific directories, causing errors in OpenCV's file reading.
- **Solution**: The file paths were parsed to extract the filename, and `MEDIA_ROOT` was used to ensure correct path handling within Django.

### 2. Edge Detection Complexity

- **Issue**: Applying both Canny and DoG with morphology in real-time was initially challenging.
- **Solution**: A UI slider was implemented to dynamically adjust kernel size for the DoG edge detection, enhancing flexibility for the user.

### 3. Human Detection Model Integration

- **Issue**: Integrating YOLOv3 into the pipeline required careful handling of the detection model and confidence filtering.
- **Solution**: Pre-trained YOLOv3 weights were loaded, and a confidence threshold was applied to filter out low-confidence detections.

# 6. Conclusion

The project offers a comprehensive solution for image stitching, edge detection, and AI-based human detection. It seamlessly integrates various image processing techniques and provides a user-friendly interface that allows dynamic adjustments for certain operations. 

This system can be extended to other types of object detection or image analysis tasks, making it versatile for various use cases.
