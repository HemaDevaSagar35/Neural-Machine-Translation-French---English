import os
import h5py
import pickle
import random
import numpy as np
import nmt_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Embedding
from keras.layers import Dense, Activation, Lambda, RepeatVector, Add
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD
import keras.backend as K
from datetime import datetime, timedelta


def data_fetch():
	#need to written according the choice of translation
	return source_data, target_data

def model_training(source_data, target_data, status = 'fresh'):
	##currently operatin gin default mode: so default values for parameter are used as per mentions in nmt_model.py
	if status == 'fresh':
		length = source_data.shape[0]:
		assert (length == target_data.shape[0])
		c0 = np.zeros((length, n_d))
		y0 = np.zeros((length, embedding_dim))
		target_data_ = np.expand_dims(target_data, axis=-1)
		
		### intialization of models
		## go with default arguments or change them
		model_train, model_enc_inf, model_dec_step = nmt_model.enc_dec_v2()
		custom_accuracy = nmt_model.custom_accuracy()
		custom_loss = nmt_model.custom_loss()
		opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model_train.compile(opt,metrics=[custom_accuracy], loss=custom_loss)
		model_train.fit([source_data, c0, y0, target_data], target_data_, epochs=10, batch_size=n_batch_size)
	
		model_train.save('model_weights.h5')
	elif status == 'continue':
		n_e = 512
		source_data = None (#this is like place holder, I can initiate source_data with new language corpus)
		target_data = None (#this is like place holder, I can initiate target_data with new Language corpus)
		length = source_data.shape[0]:
		assert (length == target_data.shape[0])
		c0 = np.zeros((length, n_d))
		y0 = np.zeros((length, embedding_dim))
		target_data_ = np.expand_dims(target_data, axis=-1)
		
		custom_accuracy = nmt_model.custom_accuracy()
		custom_loss = nmt_model.custom_loss()
		
		model_train = load_model('model_weights.h5',custom_objects = {'n_e':n_e, 'custom_loss':custom_loss, 'custom_accuracy':custom_accuracy})
		model_train.save('model_weights.h5')
	else:
		print('No Training done because status is not clealy mentioned')
		
#### some way of getting your source and target data
##source data shape : (samples, Tx)
##target data shape : (samples, Ty

if name == "__main__":
	##currently operatin gin default mode: so default values for parameter are used as per mentions in nmt_model.py
	if len(sys.argv) > 1:
		status = sys.argv[1]
	else:
		status = 'fresh'
	
	source_data, target_data = data_fetch()
	model_training(source_data, target_data, status = status)
			

	
