import os

from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # this was my best model 

# Run batched inference on a list of images

images_dir = "sample_images_for_inference"
images = os.listdir(images_dir)
images = [x for x in images if x.endswith(".jpg")]
images = [os.path.join(images_dir, x) for x in images]
results = model(images)  # return a list of Results objects

# Process results list
# the boxes and probabilities are what we will want
# feel free to comment out the result.show() and just pull the boxes however you need 
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk