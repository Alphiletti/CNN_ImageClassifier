import os
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

bs = 32

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = bs,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = bs,
                                            class_mode = 'binary')

list_train_street = os.listdir('dataset/training_set/street')
list_train_forest = os.listdir('dataset/training_set/forest')
list_test_street = os.listdir('dataset/test_set/street')
list_test_forest = os.listdir('dataset/test_set/forest')
steps = len(list_train_street) + len(list_train_forest)
validation = len(list_test_street) + len(list_test_forest)

classifier.fit_generator(training_set,
                         steps_per_epoch = steps / bs, # training data image count / batch size
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = validation / bs) # test data image count / batch size

# Part 3 - Saving Models
classifier_structure = classifier.to_json()
f = Path("classifier_structure.json")
f.write_text(classifier_structure)

classifier.save_weights("classifier_weights.h5")
