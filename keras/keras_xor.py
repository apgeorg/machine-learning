from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

# Trainset 
trX = np.array([[0, 0], 
              [0, 1],
              [1, 0],
              [1, 1]])
# Expected Output
trY = np.array([[0], 
              [1],
              [1],
              [0]])
# Testset              
teX = np.array([[0.1, 0],
                [0.2, 0.9],
                [1.1, 0.2],
                [0.8, 0.9]])
# Create model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
model.fit(trX, trY, nb_epoch=1000, batch_size=1)
# Evaluate the model
scores = model.evaluate(trX, trY)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Print predictions
predictions = model.predict(teX)
for pred in predictions:
    print ("%.3f" % (pred))
