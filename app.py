import streamlit as st
from streamlit_cropper import st_cropper
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image, ImageFile
from random import sample
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
from pathlib import Path
Image.LOAD_TRUNCATED_IMAGES = True
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import os
import pickle

listing_data = pd.read_csv("./farfetch-listings/current_farfetch_listings.csv")
listing_data.drop('Unnamed: 0', axis=1, inplace=True)
listing_data.drop('priceInfo.installmentsLabel', axis=1, inplace=True)
listing_data.drop('merchandiseLabel', axis=1, inplace=True)
listing_data['priceInfo.discountLabel'] = listing_data['priceInfo.discountLabel'].fillna(0)
listing_data.drop('availableSizes', axis=1, inplace=True)
# Store the directory path in a varaible
cutout_img_dir = "./farfetch-listings/cutout-img/cutout"
model_img_dir = "./farfetch-listings/model-img/model"

# list the directories
cutout_images = os.listdir(cutout_img_dir)
model_images = os.listdir(model_img_dir)

def extractImageName(x):
    
    # 1. Invert the image path
    x_inv = x[ :: -1]
    
    # 2. Find the index of '/'
    slash_idx = x_inv.find('/')
    
    # 3. Extract the text after the -slash_idx
    return x[-slash_idx : ] 

listing_data['cutOutimageNames'] = listing_data['images.cutOut'].apply(lambda x : extractImageName(x))
listing_data['modelimageNames'] = listing_data['images.model'].apply(lambda x : extractImageName(x))

# Extract only those data points for which we have images
listing_data = listing_data[listing_data['cutOutimageNames'].isin(cutout_images)]
listing_data = listing_data[listing_data['modelimageNames'].isin(model_images)]
# Reset the index
listing_data.reset_index(drop=True, inplace=True)

# Add entire paths to cutOut and modelImages
listing_data['cutOutImages_path'] = cutout_img_dir + '/' + listing_data['cutOutimageNames']
listing_data['modelImages_path'] = model_img_dir + '/' + listing_data['modelimageNames']
# Drop the cutOutimageNames, cutOutimageNames
listing_data.drop(['cutOutimageNames', 'cutOutimageNames'], axis=1, inplace=True)


class FeatureExtractor:
    
    # Constructor
    def __init__(self, arch='VGG'):
        
        self.arch = arch
        
        if self.arch == 'ResNet':
            base_model = ResNet50(weights = 'imagenet')
            self.model = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)
            
    
    # Method to extract image features
    def extract_features(self, img):
        
        # The VGG 16 & ResNet 50 model has images of 224,244 as input while the Xception has 299, 299
        if self.arch == 'ResNet':
            img = tf.image.resize(img, (224, 224))
        
        # Ensure the image is in the range [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        # Remove the batch dimension
        img = tf.squeeze(img, axis=0)
        
        # Convert the tensor to a NumPy array
        x = image.img_to_array(img)

        x = np.copy(x)
        
        # Expand dimensions to make it (1, 224, 224, 3) if needed
        x = np.expand_dims(x, axis=0)
    
            
        if self.arch == 'ResNet':
            # Proprocess the input as per ResNet 50
            x = resnet_preprocess(x)
        
        # Extract the features
        features = self.model.predict(x) 
        
        # Scale the features
        features = features / np.linalg.norm(features)
        
        return features 
    
resnet_feature_extractor = FeatureExtractor(arch='ResNet')

with open('farfetch_embs_50k.pkl', 'rb') as f:
    image_features_resnet = pickle.load(f)

def compute_similarity(query_features, image_features):
    """
    Compute similarity between the query features and all image features using Euclidean distance.
    """
    similarity_images = {}
    for idx, feat in image_features.items():
        # Compute the similarity using Euclidean Distance
        similarity_images[idx] = np.sum((query_features - feat) ** 2) ** 0.5
    return similarity_images

model_path = "./models/attire_classication2.h5"

model = tf.keras.models.load_model(model_path)

unique_labels = np.array(['Accessories', 'Apparel Set', 'Bags', 'Bath and Body',
       'Beauty Accessories', 'Belts', 'Bottomwear', 'Cufflinks', 'Dress',
       'Eyes', 'Eyewear', 'Flip Flops', 'Fragrance', 'Free Gifts',
       'Gloves', 'Hair', 'Headwear', 'Home Furnishing', 'Innerwear',
       'Jewellery', 'Lips', 'Loungewear and Nightwear', 'Makeup',
       'Mufflers', 'Nails', 'Perfumes', 'Sandal', 'Saree', 'Scarves',
       'Shoe Accessories', 'Shoes', 'Skin', 'Skin Care', 'Socks',
       'Sports Accessories', 'Sports Equipment', 'Stoles', 'Ties',
       'Topwear', 'Umbrellas', 'Vouchers', 'Wallets', 'Watches',
       'Water Bottle', 'Wristbands'], dtype=object)
# Define image size
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label

# Define the batch size, 32 is a good default
BATCH_SIZE = 32
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch

def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_labels[np.argmax(prediction_probabilities)]

def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_labels[np.argmax(label)])
  return images, labels

def preprocess_cropped_image(cropped_image):
    """
    Preprocess the cropped image for model prediction.
    """
    # Convert the PIL image to a TensorFlow tensor
    cropped_image = tf.convert_to_tensor(np.array(cropped_image))
    cropped_image = tf.image.resize(cropped_image, size=[IMG_SIZE, IMG_SIZE])
    cropped_image = tf.expand_dims(cropped_image, axis=0)  # Expand dimensions to create a batch of 1
    cropped_image = cropped_image / 255.0  # Normalize the image
    return cropped_image

# Define a function to make a prediction on the cropped image
def predict_attire(cropped_img):
    """
    Make a prediction on the cropped image using the loaded model.
    """
    # Preprocess the cropped image
    processed_cropped_img = preprocess_cropped_image(cropped_img)
    
    # Make predictions using the model
    predictions = model.predict(processed_cropped_img)
    
    # Get the predicted label
    predicted_label = get_pred_label(predictions)
    
    # Display the predicted label in the Streamlit app
    st.write(f"Predicted Attire: {predicted_label}")

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
stroke_width = st.sidebar.number_input(label="Box Thickness", value=3, step=1)




aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

return_type_choice = st.sidebar.radio(label="Return type", options=["Cropped image", "Rect coords"])
return_type_dict = {
    "Cropped image": "image",
    "Rect coords": "box"
}
return_type = return_type_dict[return_type_choice]

#prediction buttons
predict_attire_button = st.sidebar.button("Predict Attire")
similiar_clothes_button = st.sidebar.button("Similiar clothes")

#setting the image to none for error 
#cropped_img = None


if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    if return_type == 'box':
        rect = st_cropper(
            img,
            realtime_update=realtime_update,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type=return_type,
            stroke_width=stroke_width
        )
        raw_image = np.asarray(img).astype('uint8')
        left, top, width, height = tuple(map(int, rect.values()))
        st.write(rect)
        masked_image = np.zeros(raw_image.shape, dtype='uint8')
        masked_image[top:top + height, left:left + width] = raw_image[top:top + height, left:left + width]
        st.image(Image.fromarray(masked_image), caption='masked image')
    else:
        # Get a cropped image from the frontend
        cropped_img = st_cropper(
            img,
            realtime_update=realtime_update,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type=return_type,
            stroke_width=stroke_width
        )

        temp_cropped_path = "./temp_cropped_image.jpg"
        cropped_img.save(temp_cropped_path)

        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((224, 224))
        st.image(cropped_img)

if predict_attire_button:
    if cropped_img is not None :
        with st.spinner("Loading..."):
            time.sleep(2)
            
            temp_cropped_path = "./temp_cropped_image.jpg"

            saved_cropped_img = Image.open(temp_cropped_path)
            predict_attire(saved_cropped_img)

    else : st.write('❌ Choose an Image first ! ❌')

if similiar_clothes_button:
   if cropped_img is not None:
      with st.spinner("Loading..."):
         time.sleep(2)
         temp_cropped_path = "./temp_cropped_image.jpg"
         processed_cropped_img = preprocess_cropped_image(cropped_img)
            
         # Extract features for the cropped image using ResNet
         query_features_Resnet = resnet_feature_extractor.extract_features(processed_cropped_img)
         
         # Compute similarity between the query features and all image features using Euclidean distance
         similarity_images_resnet = compute_similarity(query_features_Resnet, image_features_resnet)
         
         # Sort the similarities and get top 10 similar images
         similarity_resnet_sorted = sorted(similarity_images_resnet.items(), key=lambda x: x[1], reverse=False)
         top_10_indexes_resnet = [idx for idx, _ in similarity_resnet_sorted][:10]
         
         # Display the top 10 most similar images
         st.write("Similar Clothes:")
         for idx in top_10_indexes_resnet:
             similar_image_path = listing_data.iloc[idx]['cutOutImages_path']
             similar_img = Image.open(similar_image_path)
             
             # Display the similar image
             st.image(similar_img, caption=f"Similar Image {idx+1}")

   else : st.write('❌ Choose an Image first ! ❌')
