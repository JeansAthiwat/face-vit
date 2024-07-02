import os
from facenet_pytorch import MTCNN
from PIL import Image

# Define the directories
input_dir = "/home/jeans/internship/resources/datasets/lfw"  # Replace with your LFW dataset directory
output_dir = "/home/jeans/internship/resources/datasets/lfw-MTCNN-128x128"  # Replace with your desired output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize the MTCNN model
mtcnn = MTCNN(image_size=128, margin=0, thresholds=[0.4, 0.5, 0.5])

# Process each image in the input directory
for root, dirs, files in os.walk(input_dir):
    # Create corresponding output directory structure
    rel_path = os.path.relpath(root, start=input_dir)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(out_path, exist_ok=True)

    for file in files:
        if file.endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(root, file)
            img = Image.open(img_path)

            # Detect and crop face
            boxes, _ = mtcnn.detect(img)

            # Check if a face is detected
            if boxes is not None:
                # Use the first detected face
                box = boxes[0]
                left, top, right, bottom = map(int, box)
                cropped_img = img.crop((left, top, right, bottom))
                cropped_img = cropped_img.resize((128, 128), Image.LANCZOS)
            else:
                # If no face is detected, resize the original image
                print("no face detected use origin resized")
                cropped_img = img.resize((128, 128), Image.LANCZOS)

            # Save the processed image to corresponding output directory
            out_file = os.path.join(out_path, file)
            print(f"saved: {out_file}")
            cropped_img.save(out_file)
