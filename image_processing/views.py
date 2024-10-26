from django.shortcuts import render, redirect, reverse
from .forms import ImageUploadForm, SmoothingForm, BoxForm, DogForm, MultiImageUploadForm
from .operation import apply_smoothing, apply_box, apply_dog, apply_canny, apply_stitching
from django.conf import settings
import os
import cv2
import ast
import numpy as np

def home(request):
    
    return render(request, 'image_processing/home.html')

def about(request):
    return render(request, 'image_processing/about.html', {'title': 'About'})

def box(request):
    if request.method == 'POST':
        image_form = ImageUploadForm(request.POST, request.FILES)
        box_form = BoxForm(request.POST)
        if image_form.is_valid() and box_form.is_valid():
            image = image_form.cleaned_data['image']
            mask_size = box_form.cleaned_data['mask_size']
            # Ensure media directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            # Save the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Apply smoothing operation
            result_image_path = apply_box(image_path, mask_size)

            return redirect(reverse('box_result', kwargs={'result_image': result_image_path}))

    else:
        image_form = ImageUploadForm()
        box_form = BoxForm()

    return render(request, 'image_processing/box.html', {'image_form': image_form, 'box_form': box_form})


def box_result(request, result_image):

    if isinstance(result_image, str):
        result_image = ast.literal_eval(result_image)

    result_image1 = os.path.join(settings.MEDIA_URL, result_image[0])  # Resized image path
    result_image2 = os.path.join(settings.MEDIA_URL, result_image[1])

    return render(request, 'image_processing/box_result.html', {
        'result_image1': result_image1,
        'result_image2': result_image2,
    })


def smoothing(request):
    if request.method == 'POST':
        image_form = ImageUploadForm(request.POST, request.FILES)
        smoothing_form = SmoothingForm(request.POST)
        if image_form.is_valid() and smoothing_form.is_valid():
            image = image_form.cleaned_data['image']
            mask_size = smoothing_form.cleaned_data['mask_size']
            # Ensure media directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            # Save the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Apply smoothing operation
            result_image_path = apply_smoothing(image_path, mask_size)

            return redirect(reverse('smoothing_result', kwargs={'result_image': result_image_path}))

    else:
        image_form = ImageUploadForm()
        smoothing_form = SmoothingForm()

    return render(request, 'image_processing/smoothing.html', {'image_form': image_form, 'smoothing_form': smoothing_form})

def smoothing_result(request, result_image):

    if isinstance(result_image, str):
        result_image = ast.literal_eval(result_image)

    result_image1 = os.path.join(settings.MEDIA_URL, result_image[0])  # Resized image path
    result_image2 = os.path.join(settings.MEDIA_URL, result_image[1])

    return render(request, 'image_processing/smoothing_result.html', {
        'result_image1': result_image1,
        'result_image2': result_image2,
    })


def dog(request):
    if request.method == 'POST':
        image_form = ImageUploadForm(request.POST, request.FILES)
        dog_form = DogForm(request.POST)
        if image_form.is_valid() and dog_form.is_valid():
            image = image_form.cleaned_data['image']
            mask_size1 = dog_form.cleaned_data['mask_size1']
            mask_size2 = dog_form.cleaned_data['mask_size2']
            # Ensure media directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            # Save the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Apply DOG operation
            result_image_path = apply_dog(image_path, mask_size1, mask_size2)

            return redirect(reverse('dog_result', kwargs={'result_image': result_image_path}))

    else:
        image_form = ImageUploadForm()
        dog_form = DogForm()
        

    return render(request, 'image_processing/dog.html', {'image_form': image_form, 'dog_form': dog_form})


def dog_result(request, result_image):

    if isinstance(result_image, str):
        result_image = ast.literal_eval(result_image)

    result_image1 = os.path.join(settings.MEDIA_URL, result_image[0])  # Resized image path
    result_image2 = os.path.join(settings.MEDIA_URL, result_image[1])

    return render(request, 'image_processing/dog_result.html', {
        'result_image1': result_image1,
        'result_image2': result_image2,
    })



def canny(request):
    if request.method == 'POST':
        image_form = ImageUploadForm(request.POST, request.FILES)
        if image_form.is_valid():
            image = image_form.cleaned_data['image']
            # Ensure media directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            # Save the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Apply canny operation
            result_image_path = apply_canny(image_path)

            return redirect(reverse('canny_result', kwargs={'result_image': result_image_path}))

    else:
        image_form = ImageUploadForm()
        

    return render(request, 'image_processing/canny.html', {'image_form': image_form})


def canny_result(request, result_image):

    if isinstance(result_image, str):
        result_image = ast.literal_eval(result_image)

    result_image1 = os.path.join(settings.MEDIA_URL, result_image[0])  # Resized image path
    result_image2 = os.path.join(settings.MEDIA_URL, result_image[1])

    return render(request, 'image_processing/canny_result.html', {
        'result_image1': result_image1,
        'result_image2': result_image2,
    })



def upload_and_stitch_images(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        image_paths = []

        # Save the uploaded images
        for image in images:
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            image_paths.append(image_path)
        
        # Call the image stitching function
        original_images, result_image = apply_stitching(image_paths)
        original_images_str = ','.join(original_images)

        return redirect(f"/stitch/result/?original_images={original_images_str}&result_image={result_image}")




    return render(request, 'image_processing/stitching.html')


def stitch_result(request):
    original_images_str = request.GET.get('original_images', '')
    result_image = request.GET.get('result_image', '')

    # Split the string back into a list
    original_images = original_images_str.split(',') if original_images_str else []

    return render(request, 'image_processing/stitching_result.html', {
        'result_image': result_image,
        'originals' : original_images,
        'MEDIA_URL' : settings.MEDIA_URL
        
    })



def process_images(request):
    if request.method == 'POST':
        # Get uploaded images from the request
        uploaded_images = request.FILES.getlist('images')

        # Save the images to MEDIA_ROOT
        image_paths = []
        for image in uploaded_images:
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            image_paths.append(image_path)

        # Stitch the images together
        original, stitched_image = apply_stitching(image_paths)
        name_stitched = os.path.join(settings.MEDIA_ROOT, stitched_image)
        stitched_image_r = cv2.imread(name_stitched)
        # Save the stitched image
        stitched_image_path = os.path.join(settings.MEDIA_ROOT, 'stitched_result.jpg')
        cv2.imwrite(stitched_image_path, stitched_image_r)

        # Apply Canny edge detection to the stitched image
        canny_edges = apply_canny(stitched_image)
        canny_edges_name = os.path.join(settings.MEDIA_ROOT, canny_edges[1])
        canny_edges_r = cv2.imread(canny_edges_name)

        # Apply DoG edge detection to the stitched image
        dog_edges = apply_dog(stitched_image, 2, 5)
        dog_edges_name = os.path.join(settings.MEDIA_ROOT, dog_edges[1])
        dog_edges_r = cv2.imread(dog_edges_name)

        # Save the results for display later
        canny_result_path = os.path.join(settings.MEDIA_ROOT, 'canny_result.jpg')
        cv2.imwrite(canny_result_path, canny_edges_r)

        dog_result_path = os.path.join(settings.MEDIA_ROOT, 'dog_result.jpg')
        cv2.imwrite(dog_result_path, dog_edges_r)

        return redirect('show_results')

    return redirect('cv-home')


def show_results(request):
    return render(request, 'image_processing/results.html')

def apply_morphological_operation(request):
    if request.method == 'POST':
        # Get the selected kernel size from the form
        kernel_size = int(request.POST.get('kernel_size', 3))

        # Load the stitched image from MEDIA_ROOT
        stitched_image_path = os.path.join(settings.MEDIA_ROOT, 'stitched_result.jpg')
        stitched_image = cv2.imread(stitched_image_path, cv2.IMREAD_GRAYSCALE)

        # Create a kernel based on the selected size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply the morphological operation (e.g., dilation)
        morphed_image = cv2.morphologyEx(stitched_image, cv2.MORPH_OPEN, kernel)

        # Save the result to display it later
        morphed_image_path = os.path.join(settings.MEDIA_ROOT, 'morphed_result.jpg')
        cv2.imwrite(morphed_image_path, morphed_image)

        # Redirect to a results page to show the morphed image
        return redirect('show_morphological_results')

    return redirect('cv-home')


def show_morphological_results(request):
    return render(request, 'image_processing/morphological_results.html')


def process(request):
    return render(request, 'image_processing/process.html')



def detect_humans(image_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolo_model/yolov3.weights", "yolo_model/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load COCO class labels
    with open("yolo_model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the stitched image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for the model (blob)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the detection results
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Loop through the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for humans and confidence level above 50%
            if classes[class_id] == "person" and confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordinates for the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to filter overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw bounding boxes around detected humans
    if isinstance(indexes, (list, tuple)):
        indexes = np.array(indexes)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw rectangle and label for human detection
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the resulting image with human detections
    human_detection_path = os.path.join(settings.MEDIA_ROOT, 'human_detection_result.jpg')
    cv2.imwrite(human_detection_path, image)

    return human_detection_path




def ai_human_detection_view(request):

    # Load the stitched image path
    stitched_image_path = os.path.join(settings.MEDIA_ROOT, 'stitched_result.jpg')

    # Perform AI-based human detection
    human_detection_image_path = detect_humans(stitched_image_path)

    # Redirect to the results page where the human detection image will be displayed
    return redirect('show_human_detection_results')


def show_human_detection_results(request):
    return render(request, 'image_processing/human_detection_results.html')

def ai_human_detection(request):
    if request.method == 'POST':
        image_form = ImageUploadForm(request.POST, request.FILES)
        if image_form.is_valid():
            image = image_form.cleaned_data['image']
            # Ensure media directory exists
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            # Save the uploaded image
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Apply human detection operation
            result_image_path = detect_humans(image_path)

            return redirect('show_human_detection_results')

    else:
        image_form = ImageUploadForm()
        

    return render(request, 'image_processing/human_detect.html', {'image_form': image_form})




   


