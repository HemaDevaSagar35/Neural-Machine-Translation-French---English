import os
import h5py
import pickle
import random
import numpy as np
import nmt_model
from nltk.translate.bleu_score import sentence_bleu
from keras.models import Model
from keras.models import load_model
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Embedding
from keras.layers import Dense, Activation, Lambda, RepeatVector, Add
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD
import keras.backend as K
from datetime import datetime, timedelta


def data_fetch():
	#need to written according the choice of translation that need to be predicted
	return source_data

def inference_fetch(source_data, model_enc_inf, model_dec_step, beam_width=3):
	##total code
	beam_size= beam_width
	total_output_with_beam = []
	total_cummulative = []
	for i in range(source_data.shape[0]):

  
		final_set = []
		cummulative_final_set = []
		temp_source = source_data[i,:].reshape((1,Tx))
		[enc_out, atten_state, s,c, dec_output] = model_enc_inf.predict([temp_source, c0, y0])
		step_output = np.argsort(dec_output,axis=1)[:,::-1][:,:beam_size]
		cummulative_proba = np.sort(dec_output,axis=1)[:,::-1][:,:beam_size]
		final_set = step_output.copy()
		s_beam_size = np.repeat(s, beam_size, axis=0)
		c_beam_size = np.repeat(c, beam_size, axis=0)


		##same as above in the for loop
		final_temp = []
		residual_beam_size = 3
		for ter in range(1,Ty):
			#print(ter)
			#print(ter)
			s = []
			c = []
			loop_out = []  
  
			for j in range(residual_beam_size):
			#print(s_beam_size.shape)
			s_temp = s_beam_size[j,:].reshape((1,n_d))
			c_temp = c_beam_size[j,:].reshape((1,n_d))
			step_output_temp = step_output[:,j].reshape((1,1))
			[s_,c_, dec_output] = model_dec_step.predict([enc_out, atten_state, s_temp,c_temp,step_output_temp])
			s.append(s_)
			c.append(c_)
			dec_output = dec_output * cummulative_proba[:,j]
			loop_out.append(dec_output)
		loop_out = np.concatenate(loop_out,axis=1)

		s = np.concatenate(s, axis=0)
		c = np.concatenate(c,axis=0)
		step_output = np.argsort(loop_out)[:,::-1][:,:residual_beam_size] % vocab_size  #:3 is hard coded 
		previous_order = (np.argsort(loop_out)[:,::-1][:,:residual_beam_size] / vocab_size).astype(np.int)
		cummulative_proba = (np.sort(loop_out)[:,::-1][:,:residual_beam_size])
		final_set = final_set[:,previous_order.ravel()]
		final_set = np.concatenate([final_set,step_output],axis=0)
		s_beam_size = s.copy()
		c_beam_size = c.copy()
  
		##here i need to implement EOS too
		##07/25
		#   if ter == 19:
		#     break
		deletion = np.where(step_output == 2)
		if np.size(deletion) != 0:
		#print('entered')
			deletion = deletion[1]
			step_output = np.delete(step_output, deletion, axis=1)
			residual_beam_size =  step_output.shape[1]
			s_beam_size = np.delete(s_beam_size, deletion, axis=0)
			c_beam_size = np.delete(c_beam_size, deletion, axis=0)
			#cummulative_proba = np.delete(cummulative_proba, deletion, axis=1)
			for dl in deletion:
				final_temp.append(final_set[:,dl])
				#print(deletion)
				#print(cummulative_proba)
				if np.size(cummulative_proba) == 0:
					print(yes)
				cummulative_final_set.extend(list(cummulative_proba[:,dl]))
			final_set = np.delete(final_set, deletion, axis=1)
			cummulative_proba = np.delete(cummulative_proba, deletion, axis=1)
      #residual_beam_size = residual_beam_size - 1
  
		if np.size(step_output) == 0:
			break
		#print(cummulative_proba)
		#print(final_set.T)
		final_temp.extend(list(final_set.T))
		#if len(cummulative_proba)!=0:
		cummulative_final_set.extend(list(cummulative_proba.ravel().T))
		total_output_with_beam.append(final_temp)
		total_cummulative.append(cummulative_final_set)

	return total_output_with_beam, total_cummulative

if name == "__main__":
	
	source_data = data_fetch()
	model_train, model_enc_inf, model_dec_step = nmt_model.enc_dec_v2()
	model_train.load_weights('trained_model_weights.h5')
	
	total_output_with_beam, total_cummulative= inference_fetch(source_data, model_enc_inf, model_dec_step, beam_width=3)
	
	###both output can be saved for later use
	
	
	