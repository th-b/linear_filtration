import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.__version__)

import numpy as np
from tqdm import tqdm
import imageio

NUM_MODELS = 100

PRED_DIR = 'predictions_att'

INPUT_SHAPE = (112, 92, 1)
NUM_CLASSES = 40

OPT = 'adam'

data_dir = 'att_faces/'

face_list = []
label_list = []

for (d,label) in zip(os.listdir(data_dir),range(40)):
    for f in os.listdir(os.path.join(data_dir, d)):
        img = np.stack([imageio.imread(os.path.join(data_dir, d, f)) / 255.0], axis=-1)
        face_list.append(img)
        label_list.append(label)
    label += 1

faces = np.stack(face_list)
labels = np.stack(label_list)

def oh(ys,n):
	return np.stack([
		np.arange(n) == y for y in ys
		]) * 1.0


x_train = faces
x_test = faces

y_train = labels
y_test = labels

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
	('att_res_full', [], [0,1,2,3]),
	#('att_res_del0134', [0,1,2,3], []),
]

def res_block(res_in, channels):		
		x = tf.keras.layers.Conv2D(channels, (3,3), strides=2, padding='same')(res_in)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		
		x = tf.keras.layers.Conv2D(channels, (3,3), padding='same')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		
		y = res_in
		y = tf.keras.layers.Conv2D(channels, (3,3), strides=2, padding=('same'))(y)
		y = tf.keras.layers.BatchNormalization()(y)
				
		out = tf.keras.layers.Add()([x,y])
		out = tf.keras.layers.Activation('relu')(out)
		
		return out

for (NAME, TRAIN_FILTER, TRANSFER_FILTER) in A:

	print(NAME)
	
	data = filter_data(TRAIN_FILTER)
	if TRANSFER_FILTER:
		data_transfer = filter_data(TRANSFER_FILTER)

	for i in tqdm(range(NUM_MODELS)):
			
		input = tf.keras.layers.Input(shape=INPUT_SHAPE)
				
		x = tf.keras.layers.Conv2D(8, (5,5), padding='same')(input)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		
		for e in range(4,9):
			channels = 2**e
			x = res_block(x, channels)
				
		x = tf.keras.layers.GlobalMaxPooling2D()(x)		
		presm = tf.keras.layers.Dense(NUM_CLASSES-len(TRAIN_FILTER))(x)
		output = tf.keras.layers.Activation('softmax')(presm)

		model = tf.keras.models.Model(input, output)

		model.compile(optimizer=OPT,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
		
		# model.summary()
			
		model.fit(data[0], data[2],
			epochs=15,
			batch_size=8,
			# validation_data=(data[3], data[5]),
			verbose=0
		)
		
		presm_model = tf.keras.models.Model(input, presm)		
		test_pred = presm_model.predict(x_test)
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
				epochs=15,
				batch_size=8,
				validation_data=(data_transfer[3], data_transfer[5]),
				verbose=0
			)
        
			presm_model_transfer = tf.keras.models.Model(input, presm_transfer)

			test_pred = presm_model_transfer.predict(x_test)		
			np.save(test_pred_path(NAME+'_transfer', i, PRED_DIR), test_pred)
	
	test_preds = np.stack([
		np.load(test_pred_path(NAME, i, PRED_DIR)) for i in range(NUM_MODELS)
	])

	np.save(test_pred_path_total(NAME, PRED_DIR), test_preds)
