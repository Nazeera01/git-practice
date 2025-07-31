import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 

# xor data set 
X1=np.array([[0,0],[0,1],[1,0],[1,1]]) #input data 
Y1=np.array([[0],[1],[1],[0]]) # output data (target)

# Initialize the model
model = Sequential()

# add the first hidden layer 4 neurons
model.add(Dense(units=4, activation='relu', input_shape=(2,)))

# add the second hidden layer 4 neurons
model.add(Dense(units=4, activation='relu'))

# add output layer with 1 neuron and sigmoid activation
model.add(Dense(units=1, activation='sigmoid'))


# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X1, Y1, epochs=500,verbose=1)

# Evaluvate the model
accuracy = model.evaluate(X1, Y1, verbose=0)[1]
print(f"Accuracy on XOR problem: {accuracy * 100:.2f}%")