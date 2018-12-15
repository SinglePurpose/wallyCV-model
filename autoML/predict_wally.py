from google.cloud import automl_v1beta1 as automl
import cv2
import sys
import os
import math

def crop_image(imagePath, size):
  image = cv2.imread(imagePath)
  #print (image.shape)
  
  # cropped = image[0:100, 2400:2500]
  
  xTiles = math.ceil(image.shape[1] / size)
  yTiles = math.ceil(image.shape[0] / size)
  imageAmount = xTiles * yTiles
  imageHeight = image.shape[0]
  imageWidth = image.shape[1]
  
  i = 0
  for xTile in range(0, xTiles):
    for yTile in range(0, yTiles):
      croppedImage = image[yTile * size:yTile * size + size, xTile * size:xTile * size + size]
      i += 1
      cv2.imwrite("cropped_images/crop%d.jpg" % i, croppedImage)

crop_image('full.jpg', 256)

project_id = 'wallycv-tom'
compute_region = 'us-central1'
model_id = 'ICN2382104611694121566'
file_path = ['test_images/1.jpg', 'test_images/2.jpg', 'test_images/3.jpg']
score_threshold = '0.0'

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
  print("Prediction results:")
  for result in response.payload:
      print("Predicted class name: {}".format(result.display_name))
      print("Predicted class score: {}".format(result.classification.score))

#for image in file_path:
	#get_prediction(image)