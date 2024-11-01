from ultralytics import YOLO
import os


# Load the trained model
model = YOLO(r'C:\Users\harjo\OneDrive\Desktop\trainmodeliat\runs\detect\yolov8-waste6\weights\best.pt')

test_images_dir = 'data/test/images'

# Loop through each image in the test directory
for image_name in os.listdir(test_images_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
        image_path = os.path.join(test_images_dir, image_name)
        results = model(image_path)  # Perform inference on the image

        # Print the results
        print(f"Results for {image_name}:")
        if results:
            for result in results:
                # Print bounding boxes and scores
                print(result.boxes)  # Or use result.show() to visualize results
        else:
            print("No detections.")