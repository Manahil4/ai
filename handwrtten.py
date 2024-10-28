import os
import cv2
import numpy as np
# matplotlib; visualing of actual handwritten digits
import matplotlib.pyplot as plt 
# tensorflow; for ml part
import tensorflow as tf
#to get datasets, load it directly from tensorflow, don't need to download csv file and 
mnist=tf.keras.datasets.mnist
#data is labelled, as in supervised learning where output is tied with input, training data to train model and testing data is to test model but this data is not given before to model and both are label below is division for test and train ;mnist dataset have already split into training and test data
#x_train is pixel data, image itslef and y train is classification that is what is written no, or alphabet
(x_train, y_train),(x_test,y_test) = mnist.load_data()
# #for grayscale rgb value is 0-255, then normalize  it to scaling to 0-1
# #data prepossesing
x_train= tf.keras.utils.normalize(x_train, axis=1) 
x_train= tf.keras.utils.normalize(x_train, axis=1) 
# #basic sequential neural network, using already build model, not talking about sequential models, here no theroy here 
# model= tf.keras.models.Sequential()
# #adding layers to this models flatten input shape 28 *28=784 pixels in 1 line, not a 2d matrix od 28 pixels
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# #dense layer , one neuron is connected with other neuron of other layrer; basic neural network layer 
# #128 units
# #activatibon function can be tf.nn

# #choose str in activation ,relu; rectify linear unit , 0 to straight 1 
# model.add(tf.keras.layers.Dense(128, activation='relu' ))
# model.add(tf.keras.layers.Dense(128, activation='relu' ))
# #output layer, 10 units separately ,
# #softmax, output o f all neuron sum up to 1, each have value between 0 and 1, it shows what digit input imag contains
# model.add(tf.keras.layers.Dense(10, activation='softmax' ))

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# # now model compile, now train model
# model.fit(x_train, y_train, epochs= 3)

# #save 
# model.save('handwritten.keras')

#now testing data and model trained above
model = tf.keras.models.load_model('handwritten.keras')
#below is test on keras dataset and the last one is testing on data provided by me
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
picno=1
while os.path.isfile(f'digit/digit{picno}.png'):
    try:        
        img = cv2.imread(f'digit/digit{picno}.png')[:,:,0]
        img = cv2.resize(img, (28, 28))
        
        img = np.invert(np.array([img]))
        prediction =model.predict(img) 
        print(f'this digit is probably a {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        if img is None:
            print(f"Image {picno} could not be loaded.")
            continue
    except Exception as e :
        print(f'Error: {e}')
    finally:
        picno+=1        
