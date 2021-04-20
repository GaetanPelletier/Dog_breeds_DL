import tensorflow.keras as keras
import gradio

#------------------#

# Load the model
model_xception_120_dog_breeds = keras.models.load_model("Model/transfert_learning_xception_full_19042021.h5")

#Load the labels
labels_120_dog_breeds = []
with open("Labels/labels_120_dog_breeds.txt", "r") as f:
  for line in f:
    labels_120_dog_breeds.append(line.strip())

# Image examples
doberman = ["Image_examples/n02107142_14066.jpg"]
labrador = ["Image_examples/n02099712_610.jpg"]
husky = ["Image_examples/n02110185_11626.jpg"]

#------------------#

# Function for preprocessing an image and predicting the dog breed
def classify_image(image_):
  img = image_.reshape((-1, 299, 299, 3))
  img = keras.applications.xception.preprocess_input(img)
  
  prediction = model_xception_120_dog_breeds.predict(img).flatten()

  return {labels_120_dog_breeds[i]: float(prediction[i]) for i in range(120)}

# Define the inputs, outputs and examples
image = gradio.inputs.Image(shape=(299, 299))
label = gradio.outputs.Label(num_top_classes=3)

sample_images = [
  doberman,
  labrador,
  husky
]

#------------------#

# Launch the application
gradio.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    title="Image classification - 120 dog breeds",
    # description="This CNN was built thanks to transfer learning (Xception, F. Chollet).",
    examples=sample_images
).launch(debug=False)
