# Model Training Lib

# Import sequential from ternsorflow.keras.models to create CNN model 
from tensorflow.keras.models import Sequential

# Import Dense, Activation and Dropout layers. Since the model will be trained on a small dataset, 
# we’ll need less number of layers than we used before
from tensorflow.keras.layers import Dense, Activation, Dropout

# Import Adam optimizer. Optimizers are used to reduce losses while training the model
from tensorflow.keras.optimizers import Adam

# import preprocess_train_data function from the python file data_preprocessing.py
from data_preprocessing import preprocess_train_data

# Create a function for defining the CNN model
def train_bot_model(train_x, train_y):

    # Define the model being sequential using the Sequential() method.
    model = Sequential()

    # The very first layer we’ll add is a Dense layer with 128 output units. 
    # Input will be the training data that is train_x and activation function as ‘relu’(Rectified Linear Unit).
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

    # Add Dropout layer with 0.5 dropout
    model.add(Dropout(0.5))

    # The second layer is also a Dense layer with 64 output units. Activation function as ‘relu’.
    model.add(Dense(64, activation='relu'))

    # Add Dropout layer with 0.5 dropout
    model.add(Dropout(0.5))

    # The last layer will be the dense layer with output units equal to the number of tags. 
    # Use softmax activation function as its last layer of our model.
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    # Fit & Save Model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)

    # save as h5 file (hierarchical format)
    model.save('chatbot_model.h5', history)
    
    print("Model File Created & Saved")


# Calling Methods to Train Model
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

