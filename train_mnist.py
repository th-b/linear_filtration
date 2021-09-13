import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.__version__)

import numpy as np
from tqdm import tqdm

import time

NUM_MODELS = 100

PRED_DIR = 'predictions_mnist'

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

OPT = 'adam'

def oh(ys,n):
	return np.stack([
		np.arange(n) == y for y in ys
		]) * 1.0

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

x_train = np.stack([x_train], axis = -1)
x_test = np.stack([x_test], axis = -1)
	
y_train_oh = oh(y_train,NUM_CLASSES)
y_test_oh = oh(y_test,NUM_CLASSES)


def filter_indices(a, classes):
	if not classes:
		return np.repeat(True, a.shape[0])
	
	return np.all([a != cl for cl in classes], axis=0)

def filter_data(classes):	
	ind = filter_indices(y_train, classes)
	
	x_train_nocl = x_train[ind]
	y_train_nocl = y_train[ind]
	y_train_oh_nocl = np.delete(y_train_oh[ind],classes,1)
		
	ind_test = filter_indices(y_test, classes)	
		
	x_test_nocl = x_test[ind_test]
	y_test_nocl = y_test[ind_test]
	y_test_oh_nocl = np.delete(y_test_oh[ind_test],classes,1)
	
	return (x_train_nocl, y_train_nocl, y_train_oh_nocl, x_test_nocl, y_test_nocl, y_test_oh_nocl)
	
def test_pred_path(name, i, dir):
	return os.path.join(dir, 'test_' + name + '_' + str(i) + '.npy')
	
def test_pred_path_total(name, dir):
	return os.path.join(dir, 'test_' + name + '.npy')


A = [
	('mnist_mlp_full', [], [0]),
	# ('mnist_mlp_del3', [3]),
	#('mnist_mlp_del7', [7),
	('mnist_mlp_del0', [0], []),
	#('mnist_mlp_del0_second', [0], []),
]

for (NAME, TRAIN_FILTER, TRANSFER_FILTER) in A:

	print(NAME)
	data = filter_data(TRAIN_FILTER)
	if TRANSFER_FILTER:
		data_transfer = filter_data(TRANSFER_FILTER)

	for i in tqdm(range(NUM_MODELS)):			
		input = tf.keras.layers.Input(shape=INPUT_SHAPE)
		x = tf.keras.layers.Flatten()(input)
		x = tf.keras.layers.Dense(50,activation='relu')(x)
		presm = tf.keras.layers.Dense(NUM_CLASSES-len(TRAIN_FILTER))(x)
		output = tf.keras.layers.Activation('softmax')(presm)
        
		model = tf.keras.models.Model(input, output)

		model.compile(optimizer=OPT,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
		
		# model.summary()
			
		model.fit(data[0], data[2],
			epochs=6,
			batch_size=32,
			# validation_data=(data[3], data[5]),
			verbose=0
		)
        
				
		presm_model = tf.keras.models.Model(input, presm)
		
		# start = time.time()			
		test_pred = presm_model.predict(x_test)
		# end = time.time()
		# print('Prediction time:', end - start)
		
		np.save(test_pred_path(NAME, i, PRED_DIR), test_pred)
        # ------------------------------------
        
		if TRANSFER_FILTER:
			presm_transfer = tf.keras.layers.Dense(NUM_CLASSES-len(TRANSFER_FILTER))(x)
			output_transfer = tf.keras.layers.Activation('softmax')(presm_transfer)
			model_transfer = tf.keras.models.Model(input, output_transfer)
        
			for l in model.layers:
				l.trainable = False
        
			model_transfer.compile(optimizer=OPT,
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
        
			model_transfer.fit(data_transfer[0], data_transfer[2],
				epochs=6,
				batch_size=32,
				# validation_data=(data[3], data[5]),
				verbose=0
			)
        
			presm_model_transfer = tf.keras.models.Model(input, presm_transfer)

			test_pred = presm_model_transfer.predict(x_test)		
			np.save(test_pred_path(NAME+'_transfer', i, PRED_DIR), test_pred)
        
		
	preds = np.stack([
		np.load(test_pred_path(NAME, i, PRED_DIR)) for i in range(NUM_MODELS)
	])
	
	np.save(test_pred_path_total(NAME, PRED_DIR), preds)
