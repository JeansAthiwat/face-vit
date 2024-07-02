import os
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms.functional as TF

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
            face_tensor = mtcnn(img)

            # Check if a face is detected
            if face_tensor is not None:
                # Convert tensor to PIL image
                cropped_img = TF.to_pil_image(face_tensor.squeeze(0))
            else:
                # If no face is detected, resize the original image
                cropped_img = img.resize((128, 128), Image.LANCZOS)

            # Save the processed image to corresponding output directory
            out_file = os.path.join(out_path, file)
            print(f"saved: {out_file}")
            cropped_img.save(out_file)
