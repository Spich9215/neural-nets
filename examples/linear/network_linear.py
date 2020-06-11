import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv('C:/Users/wspic/Documents/GitHub/neural-nets/examples/linear/data/train.csv')

# print(train_df.head())
np.random.shuffle(train_df.values) # Important to shuffle so things are not highly correlated. Shuffle in place
# print(train_df.head())
# print(train_df.x.values[0:5])
# print(type(train_df.x.values))

# Sequential allows us to list out layers in out network - Fully Connected Feed Forward Network (defined by Dense)
# 4 neurons - hidden layer 
# 2 neuron input layer (x, y)
# Activation Function is "relu"
# For binary classification use either Sigmoid or Softmax for activation function.
model = keras.Sequential([
	keras.layers.Dense(4, input_shape=(2,), activation='relu'),
	keras.layers.Dense(2, activation='sigmoid')])

# From logits = true because we have not scaled our data
# Adam Optimizer updates the NN based of the SparseCategorical CrossEntropy Function
model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	          metrics=['accuracy'])



x = np.column_stack((train_df.x.values, train_df.y.values)) # x and y columns are paired together

# x is the x,y coordinates, y i the color, batch size is arbitrary
model.fit(x, train_df.color.values, batch_size=4, epochs=5)

test_df = pd.read_csv('C:/Users/wspic/Documents/GitHub/neural-nets/examples/linear/data/train.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
model.evaluate(test_x, test_df.color.values)


