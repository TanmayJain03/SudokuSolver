# Import any ML library here (eg torch, keras, tensorflow)
# Start Editing
import tensorflow as tf
from tensorflow import keras
# End Editing

import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
from dataLoader import Loader
import os
import cv2

# (Optional) If you want to define any custom module (eg a custom pytorch module), this is the place to do so
# Start Editing
# End Editing


# This is the class for training our model
class Trainer:
	def __init__(self):

		# Seed the RNG's
		# This is the point where you seed your ML library, eg torch.manual_seed(12345)
		# Start Editing
		np.random.seed(12345)
		random.seed(12345)

		# End Editing

		# Set hyperparameters. Fiddle around with the hyperparameters as different ones can give you better results
		# (Optional) Figure out a way to do grid search on the hyperparameters to find the optimal set
		# Start Editing
		self.batch_size = 64 # Batch Size
		self.num_epochs = 20 # Number of Epochs to train for
		# self.num_epochs = 75	# For basic nn
		self.lr = 0.01       # Learning rate
		# End Editing

		# Init the model, loss, optimizer etc
		# This is the place where you define your model (the neural net architecture)
		# Experiment with different models
		# For beginners, I suggest a simple neural network with a hidden layer of size 32 (and an output layer of size 10 of course)
		# Don't forget the activation function after the hidden layer (I suggest sigmoid activation for beginners)
		# Also set an appropriate loss function. For beginners I suggest the Cross Entropy Loss
		# Also set an appropriate optimizer. For beginners go with gradient descent (SGD), but others can play around with Adam, AdaGrad and you can even try a scheduler for the learning rate
		# Start Editing
		self.model = keras.Sequential([
				keras.layers.Conv2D(
					activation = 'relu',
					filters = 32,
					kernel_size = (3,3), 
					padding = 'same',
					input_shape = (28,28,1)
					),
				keras.layers.BatchNormalization(),
				keras.layers.MaxPooling2D(),
				keras.layers.Flatten(),
				keras.layers.Dense(units = 10, activation = 'softmax')
			])
		# self.model = keras.Sequential([
		# 		keras.layers.Dense(units = 32, activation = 'relu'),
		# 		keras.layers.Dense(units = 10, activation = 'softmax')
		# 	])	# For basic nn
		self.loss = keras.losses.SparseCategoricalCrossentropy()
		self.optimizer = keras.optimizers.SGD(learning_rate = self.lr)
		# End Editing

	def load_data(self):
		# Load Data
		self.loader = Loader()

		# Change Data into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want each data point as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		# Start Editing
		self.loader.train_data = self.loader.train_data / 255.0
		self.loader.test_data = self.loader.test_data / 255.0

		self.loader.train_data = self.loader.train_data.reshape((-1,28,28,1))
		# self.loader.train_data = self.loader.train_data.reshape((-1,784))	# For basic nn
		self.loader.train_labels = self.loader.train_labels.reshape((-1,1))
		self.loader.test_data = self.loader.test_data.reshape((-1,28,28,1))
		# self.loader.test_data = self.loader.test_data.reshape((-1,784))	# For basic nn
		self.loader.test_labels = self.loader.test_labels.reshape((-1,1))


		# End Editing
		pass

	def save_model(self):
		# Save the model parameters into the file 'assets/model'
		# eg. For pytorch, torch.save(self.model.state_dict(), 'assets/model')
		# Start Editing
		self.model.save('assets/model')


		# End Editing
		pass

	def load_model(self):
		# Load the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			# eg. For pytorch, self.model.load_state_dict(torch.load('assets/model'))
			self.model = keras.models.load_model('assets/model')
			pass
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return

		training_accuracy = []
		training_loss = []
		self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = 'acc')

		print("Training...")
		for epoch in range(self.num_epochs):
			train_loss, correct = self.run_epoch()

			# For beginners, you can leave this alone as it is
			# For others, you can try out splitting the train data into train + val data, and use the validation loss to determine whether to save the model or not
			# Start Editing
			# End Editing

			print(f'	Epoch #{epoch+1} trained')
			print(f'		Train loss: {train_loss:.3f}')
			print(f'		Train Accuracy: {(correct*100):.2f}%')
			training_loss.append(train_loss)
			training_accuracy.append(correct*100)

		print('Training Complete')
		figure, axis = plt.subplots(2)
		axis[0].plot(training_accuracy, color = 'orange', marker = 'o', linewidth = 2)
		axis[0].set_title('Training Accuracy')
		axis[1].plot(training_loss, color = 'blue', marker = 'x', linewidth = 2)
		axis[1].set_title('Training Loss')
		figure.savefig('cnn.png')
		# figure.savefig('basic_nn.png')	# For basic nn
		self.model.save('assets/model')

	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initialize running loss
		running_loss = 0.0

		# Start Editing

		# Set the ML library to freeze the parameter training
		self.model.trainable = False

		i = 0 # Number of batches
		correct = 0 # Number of correct predictions
		for batch in range(0, self.loader.test_data.shape[0], self.batch_size):
			batch_X = self.loader.test_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.loader.test_labels[batch: batch+self.batch_size] # shape [batch_size,]
			batch_Y = batch_Y.reshape((-1,1))

			# Find the predictions
			predictions = self.model(batch_X, training = False)
			loss_val = self.loss(batch_Y, predictions)
			# Find the number of correct predictions and update correct
			predictions = np.argmax(np.asarray(predictions).reshape((-1,10)), axis = 1).reshape((-1,1))

			correct += np.where(predictions == batch_Y)[0].shape[0]

			# Update running_loss
			running_loss += loss_val
			i += 1
		
		# End Editing

		print(f'	Test loss: {(running_loss/i):.3f}')
		print(f'	Test accuracy: {(correct*100/self.loader.test_data.shape[0]):.2f}%')

		return correct/self.loader.test_data.shape[0]

	def run_epoch(self):
		# Initialize running loss
		running_loss = 0.0
		correct = 0

		# Start Editing

		# Set the ML library to enable the parameter training
		self.model.trainable = True

		# Shuffle the data (make sure to shuffle the train data in the same permutation as the train labels)
		self.loader.train_data = self.loader.train_data.reshape((-1,784))	# Remove line for basic nn
		training_data = np.concatenate((self.loader.train_data, self.loader.train_labels), axis = 1)
		np.random.shuffle(training_data)
		self.loader.train_data = training_data[0:60000, 0:784]
		self.loader.train_labels = training_data[0:60000, 784]
		self.loader.train_data = self.loader.train_data.reshape((-1,28,28,1))	# Remove line for basic nn
		self.loader.train_labels = self.loader.train_labels.reshape((-1,1))

		
		i = 0 # Number of batches
		for batch in range(0, self.loader.train_data.shape[0], self.batch_size):
			batch_X = self.loader.train_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.loader.train_labels[batch: batch+self.batch_size] # shape [batch_size,]
			batch_Y = batch_Y.reshape((-1,1))

			# Zero out the grads for the optimizer
			with tf.GradientTape() as tape:
			# Find the predictions
				predictions = self.model(batch_X, training = True)
			# Find the loss
				loss_val = self.loss(batch_Y, predictions)
			# Backpropagation
				grads = tape.gradient(loss_val, self.model.trainable_variables)
				self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
			# Accuracy
				predictions = np.argmax(np.asarray(predictions).reshape((-1,10)), axis = 1).reshape((-1,1))
				correct += np.where(predictions == batch_Y)[0].shape[0]

			# Update the running loss
			running_loss += loss_val
			i += 1
		
		# End Editing

		return running_loss / i, correct / self.loader.train_data.shape[0]

	def predict(self, image):
		prediction = 0
		if not self.model:
			return prediction

		# Start Editing

		# Change image into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want image as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		image = image.reshape((1,28,28,1))
		# image = image.reshape((1,784))	# For basic nn
		# Predict the digit value using the model
		prediction = np.argmax(self.model.predict(image), axis = 1)
		# End Editing
		return prediction

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			pass
		i = np.random.randint(0,t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i].reshape((28,28))
		image = cv2.resize(image, (0,0), fx=16, fy=16)
		cv2.imshow('Digit', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()