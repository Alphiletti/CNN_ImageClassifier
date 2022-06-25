import keras_preprocessing.image
from pathlib import Path
import numpy as np
from keras.models import model_from_json

f = Path("classifier_structure.json")
classifier_structure = f.read_text()
classifier = model_from_json(classifier_structure)
classifier.load_weights("classifier_weights.h5")

test_image = keras_preprocessing.image.load_img('dataset/single_prediction/street_or_forest_11.jpg', target_size = (64, 64))
test_image = keras_preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'street'
else:
    prediction = 'forest'

print(prediction)