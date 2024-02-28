import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Define the sequence of colors
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink']

# Convert the sequence of colors into numerical values
color_to_num = {color: i for i, color in enumerate(colors)}
num_to_color = {i: color for i, color in enumerate(colors)}

# Define the length of the input sequence and the number of output classes
seq_length = 3
num_classes = len(colors)

# Generate input/output sequences
X = []
y = []
for i in range(len(colors) - seq_length):
    seq_in = colors[i:i+seq_length]
    seq_out = colors[i+seq_length]
    X.append([color_to_num[color] for color in seq_in])
    y.append(color_to_num[seq_out])
X = np.array(X)
y = np.array(y)

# Define the RNN model
model = Sequential()
model.add(LSTM(32, input_shape=(seq_length, 1)))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=8)

# Generate a prediction for the next color
seq = ['red', 'green', 'blue']
x_pred = np.array([color_to_num[color] for color in seq])
x_pred = np.reshape(x_pred, (1, seq_length, 1))
pred = model.predict(x_pred)[0]
next_color = num_to_color[np.argmax(pred)]

print(f"The next color in the sequence is {next_color}")
