from google.cloud import automl_v1beta1 as automl
import cv2
import sys
import os
import math
import numpy
from PIL import Image

project_id = 'wallycv-tom'
compute_region = 'us-central1'
model_id = 'ICN2382104611694121566'
score_threshold = '0.0'
results64 = []
results128 = []
results256 = []
cropped_image_paths = []

# splits full sized image into smaller parts and saves them
def crop_image(imagePath, size):
  image = cv2.imread(imagePath)
  
  xTiles = math.ceil(image.shape[1] / size)
  yTiles = math.ceil(image.shape[0] / size)
  
  i = 0
  for xTile in range(0, xTiles):
    for yTile in range(0, yTiles):
      croppedImage = image[yTile * size:yTile * size + size, xTile * size:xTile * size + size]
      i += 1
      cv2.imwrite("cropped_images/crop%d.jpg" % i, croppedImage)
      cropped_image_paths.append('cropped_images/crop' + str(i) + '.jpg')


automl_client = automl.AutoMlClient()

# Get the full path of the model.
model_full_id = automl_client.model_path(
    project_id, compute_region, model_id
)

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()

def get_prediction(image):
  # Read the image and assign to payload.
  with open(image, "rb") as image_file:
      content = image_file.read()
  payload = {"image": {"image_bytes": content}}
  
  # params is additional domain-specific parameters.
  # score_threshold is used to filter the result
  # Initialize params
  params = {}
  if score_threshold:
      params = {"score_threshold": score_threshold}
  
  response = prediction_client.predict(model_full_id, payload, params)

# generated prediction arrays for each label
  for result in response.payload:
      if result.display_name == "waldo_64":
        results64.append(result.classification.score)

      if result.display_name == "waldo_128":
        results128.append(result.classification.score)

      if result.display_name == "waldo_256":
        results256.append(result.classification.score)


crop_image('full.jpg', 256)

# starts prediction for each cropped image
j = 1
for image in cropped_image_paths:
  print(str(j) + "/" + str(len(cropped_image_paths)) + " images predicted")
  j += 1
  get_prediction(image)

# prints highest prediction index for each label
print("64 best pred index: " + str(numpy.argmax(results64)))
print("128 best pred index: " + str(numpy.argmax(results128)))
print("256 best pred index: " + str(numpy.argmax(results256)))


# searches for image with highest prediction and shows it
img = Image.open('cropped_images/crop' + str(numpy.argmax(results256) + 1) + '.jpg')
img.show()