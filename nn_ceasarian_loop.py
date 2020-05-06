# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import random

#best: elu,elu,sigmoid,12,8,binary_crossentropy, adam, 200, 10

activision_list = ['elu', 'softmax', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
loss_function_list = [		
 'mean_squared_logarithmic_error',		#this
 'logcosh', 							#this
 'binary_crossentropy', 				#this
	#  'huber_loss', 							#this
	#  'kullback_leibler_divergence',
	#  'poisson', 
	#  'squared_hinge', 
	#  'hinge', 
	#  'categorical_hinge', 
	#  'cosine_proximity'
	#  'mean_squared_error',
	#  'mean_absolute_error', 
	#  'mean_absolute_percentage_error', 
 ] 

def print_information():

	print('First activision function: %s' % first_activation)
	print('Second activision function: %s' % second_activation)
	print("Loss function: %s" % loss_function)
	print('First layer size: %d' % first_layer_size)
	print('Second layer size: %d' % second_layer_size)
	print('Number of epochs: %d' % number_of_epochs)
	print('Batch size: %d' % number_batch_size)
	print('Validation split: %.2f' % val_split)

def plot_history(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def test_set_prediction():
	count_right_predictions = 0.0
	for i in range(10):
		print('%s => %d (expected %d)' % (Xt[i].tolist(), predictions[i], yt[i]))
		if predictions[i] == yt[i]:
			count_right_predictions += 1
	print(count_right_predictions)
	print('Accuracy on test set: %.2f' % (count_right_predictions/10))

def random_activision():
	activision = random.choice(activision_list)
	return activision

def random_loss_function():
	loss_function = random.choice(loss_function_list)
	return loss_function

def switch_optimazer(arg):
	switcher = {
		#learning_rate=0.01, momentum=0.0, nesterov=False
		1: optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
		#learning rate, can be freely tuned >=0)
		2: optimizers.RMSprop(learning_rate=0.001, rho=0.9),
		#It is recommended to leave the parameters of this optimizer at their default values.
		3: optimizers.Adagrad(learning_rate=0.01),
		#It is recommended to leave the parameters of this optimizer at their default values.
		4: optimizers.Adadelta(learning_rate=1.0, rho=0.95),
		5: optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
		6: optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
		#It is recommended to leave the parameters of this optimizer at their default values.
		7: optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
	}
	return switcher.get(arg, "Invalid number")

max_acc_val=0.0

while(max_acc_val < 0.8):

    #define variables
    #ACTIVITION OPTIONS:
    #elu, softmax, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear
    # first_activation = 'elu'
    first_activation = random_activision()
    # second_activation = 'elu'
    second_activation = random_activision()

    third_activation = random_activision()
    #12,8
    # first_layer_size = 80
    first_layer_size = random.randint(10,100)
    # second_layer_size = 50
    second_layer_size = random.randint(5,50)

    #LOSS FUNCTION
    #mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error
    #squared_hinge, hinge, categorical_hinge, logcosh, huber_loss, categorical_crossentropy
    #sparse_categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence
    #poisson, cosine_proximity, is_categorical_crossentropy

    # loss_function = 'mean_squared_logarithmic_error'
    loss_function = random_loss_function()
    print(loss_function)

    #OPTIMIZERS
    # optimizer_function = switch_optimazer(5)
    optimizer_function = switch_optimazer(random.randint(1,7))

    # number_of_epochs = 1000
    number_of_epochs = random.randint(100,1000)
    # number_batch_size = 10
    number_batch_size = random.randint(5,80)

    val_split = 0.15

    # load the dataset
    dataset = loadtxt('caesarian_dataset.arff', delimiter=',')
    # dataset_train = loadtxt('caesarian_dataset_train.arff', delimiter=',')
    # dataset_test = loadtxt('caesarian_dataset_test.arff', delimiter=',')


    # split into input (X) and output (y) variables
    X = dataset[:,0:5]
    y = dataset[:,5]
    # X = dataset_train[:,0:5]
    # y = dataset_train[:,5]
    # Xt = dataset_test[:,0:5]
    # yt = dataset_test[:,5]

    # define the keras model
    model = Sequential()
    model.add(Dense(first_layer_size, input_dim=5, activation = first_activation))
    model.add(Dense(second_layer_size, activation = second_activation))
    model.add(Dense(1, activation = third_activation))

    # compile the keras model
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=['accuracy'])

    # fit the keras model on the dataset

    history = model.fit(X, y, validation_split=val_split,epochs=number_of_epochs, batch_size=number_batch_size, verbose=1)

    # list all data in history
    print(history.history.keys())
   
    # summarize history for accuracy
    # make class predictions with the model
    predictions = model.predict_classes(X)
    # summarize the first 5 cases 
    # test_set_prediction()
    max_acc_val = max(history.history['val_accuracy'])
    print('MAX_DOKLADNOSC_WALIDACJI: %.2f' % max_acc_val)
    print_information()


plot_history(history)

