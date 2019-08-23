
####------------------------####
##Author : Sagar
##references : 
#####1) https://keras.io/
#####2) https://arxiv.org/pdf/1409.0473.pdf

import os
import h5py
import pickle
import random
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Embedding
from keras.layers import Dense, Activation, Lambda, RepeatVector, Add
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD
import keras.backend as K
from datetime import datetime, timedelta



def enc_dec_v2(Tx = 30,Ty = 30, n_e = 512, n_d = 512, embedding_dim = 100, vocab_size = 30004, batch_size = None, inf_batch_size = None):
    
	'''NOTE: This function comprises all the sub components in the same place. 
	#####1) First, model is the enc-dec with attention architucture suitable for training. While training we don't pass output from one LSTM cell 
			as input to other LSTM cell in decoder. Rather the true label is passed while training. So the decoder part used while training should be 
			modified when it comes to actual translation after training.
	#####2) Second, model_enc_inf this enc-dec with attention suitable to make inference or tranlsation. This part specifically focuses on taking
			initial values of states, inputs and give the decoder's first cel outputs, states and also returns sequences of the encoder. 
	#####3) Third, model_dec_step again this enc-dec with attention slightly different from 2, and this function takes outputs from 2 and gives the 
			decoder outputs.
	
	Note : Inference model are in 2-stages because doing so makes implementing beam search (while translation) easy. A better version can be surely written
	'''
	
	'''
		Arguments
	    Tx : Input sequence length
		Ty : Output sequence length
		n_e : hidden units of LSTM cells on encoder side
		n_d : hidden units of LSTM cells on decoder side
		
	'''
	
    '''
    y0 and y = Embedding need to checked when making changes
    
    start_input: this is the tokenized input. Shape (batch size, Tx)
    c0 : This is initialization of cell state. Shape (batch size, n_d)
    y0 : This is initialization input given to Starting point of decoder. Shape (batch_size, embedding_dim)
    output_input : this is the tokenized output input. Shape (batch size, Ty)
	
	enc_output : encoder's return sequences
	atten_enc : component of attention that is dependent on encoder and static for a particular sample(looking at code while give deeper understanding)
	dec_state : decoder hidden state output
	dec_c_state : decoder cell state
	out_input : output from previous cell which is going to be one of the input to the current cell
    '''
    #inputs while training
    start_input = Input(shape=(Tx,), batch_shape=(batch_size, Tx), name = 'main')
    c0 = Input(shape=(n_d,), batch_shape=(batch_size, n_d), name='c0')
    y0 = Input(shape=(embedding_dim,), batch_shape=(batch_size, embedding_dim), name = 'y0')
    output_input = Input(shape=(Ty,), batch_shape=(batch_size, Ty), name = 'target_input')
    
    #inputs while inference
    inf_start_input = Input(shape=(Tx,), batch_shape=(inf_batch_size, Tx), name = 'inf_main')
    inf_c0 = Input(shape=(n_d,), batch_shape=(inf_batch_size, n_d), name='inf_c0')
    inf_y0 = Input(shape=(embedding_dim,), batch_shape=(inf_batch_size, embedding_dim), name = 'inf_y0')
    
    #input while inference - 2
    enc_output = Input(shape=(Tx,2*n_e), batch_shape=(inf_batch_size, Tx, 2*n_e), name = 'enc_outputs')
    atten_enc = Input(shape=(Tx,n_d), batch_shape=(inf_batch_size, Tx, n_d), name = 'atten_enc')
    dec_state = Input(shape=(n_d,), batch_shape=(inf_batch_size,n_d), name='dec_init_state')
    dec_c_state = Input(shape=(n_d,), batch_shape=(inf_batch_size, n_d), name='dec_c_state')
    out_input = Input(shape=(1,), batch_shape=(inf_batch_size,1), name='output_as_input')

    
    start_embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length = Tx, mask_zero=True)    
    encoder = Bidirectional(LSTM(n_e, return_sequences=True))
    y_embedder = Embedding(input_dim=vocab_size, output_dim = embedding_dim, mask_zero=True)
    s0_calculation = Lambda(lambda x : x[:,0,n_e:], output_shape = (n_e,), name='s0_calculation')
    s0_dense = Dense(n_d, activation='tanh')
    '''This is the step where we need attention'''
    post_layer_lstm = LSTM(n_d, return_state = True, name = 'decoder_model')
    concatenator = Concatenate(axis=-1)
    dotor = Dot(axes=1)
    repeator = RepeatVector(Tx)
    densor = Dense(n_d, name = 'attention_dense_1')
    densor_1_2 = Dense(n_d, name = 'attention_dense_12')
    activation_act_0 = Activation(activation='tanh', name = 'attention_actual')
    #print()
    densor_att_2 = Dense(1, name = 'attention_dense_2', activation='relu')
    attention_act = Activation(activation='softmax', name = 'attention_weights')
    decode_densor = Dense(vocab_size, activation='softmax', name = 'decoded_dense')
    expand_dim = Lambda(lambda x : K.expand_dims(x,axis=1))
    squeeze_dim = Lambda(lambda x : K.squeeze(x, axis=-1))
    expand_dim_last = Lambda(lambda x : K.expand_dims(x,axis=-1))
    argmax = Lambda(lambda x : K.argmax(x, axis=-1)) 
    #NEW ADDITION OF LINE
    adding_layer = Add()
    #data getting ready while training
    x = start_embed(start_input)
    x = encoder(x)    
    '''initializing hidden and cell state: What to intialize the cell state with?'''
    x2 = s0_calculation(x)
    s0 = s0_dense(x2)    
    y_ele = expand_dim(y0)    
    y = y_embedder(output_input) 
    
    s = s0
    c = c0
    s_1 = densor(x)   
    #print(K.int_shape(s_total))
    output_list = []
    for i in range(Ty):       
        s_repeat = repeator(s)
        #print(K.int_shape(s_repeat))
        s_repeat = densor_1_2(s_repeat)
        #print(K.int_shape(s_repeat))
        #s_total = concatenator([s_repeat, x])
        #s_total = densor(s_total)
        #print('shape after densor for attention {}'.format(K.int_shape(s_total)))
        s_total = adding_layer([s_1,s_repeat])
        s_total = activation_act_0(s_total)
        s_total = densor_att_2(s_total)
        s_total = squeeze_dim(s_total)
        con_weights = attention_act(s_total)
        con_weights = expand_dim_last(con_weights)
        context = dotor(inputs=[con_weights, x])
        context = concatenator([context, y_ele])
        s, _, c = post_layer_lstm(inputs=context, initial_state = [s,c])
        output = decode_densor(s)
        output = expand_dim(output)
        output_list.append(output)
        y_ele = Lambda(lambda x : x[:,i,:], output_shape = (embedding_dim,), name = 'slicing_y_prior_{}'.format(i))(y)
        y_ele = expand_dim(y_ele)
    output_final = Concatenate(axis=1)(output_list)
    print(K.int_shape(output_final))
    model = Model(inputs=[start_input,c0,y0, output_input], outputs=output_final)
    
    
    #data ready for inference
    #I am not implementing beam search(need to think about this later)
    #I am not implementing EOS stop (need to think about this, first)
    
    ##STEP - 1: SUMMARIZATION OF ENCODER AND INITIAL PREDICTION, model_enc_inf
    print('Encoder for inference is ready')
    x = start_embed(inf_start_input)
    x = encoder(x)
    '''initializing hidden and cell state: What to intialize the cell state with?'''
    x2 = s0_calculation(x)
    s0 = s0_dense(x2)   
    s_1 = densor(x)
    y_ele = expand_dim(inf_y0)
    
    #s = s0
    #c = inf_c0
   
    s_repeat = repeator(s0)
    s_repeat = densor_1_2(s_repeat)
    s_total = adding_layer([s_1,s_repeat])
    s_total = activation_act_0(s_total)
    s_total = densor_att_2(s_total)
    s_total = squeeze_dim(s_total)
    con_weights = attention_act(s_total)
    con_weights = expand_dim_last(con_weights)
    context = dotor(inputs=[con_weights, x])
    context = concatenator([context, y_ele])
    s, _, c = post_layer_lstm(inputs=context, initial_state = [s0, inf_c0])
    output = decode_densor(s)
    
    model_enc_inf = Model(inputs=[inf_start_input, inf_c0, inf_y0], outputs=[x, s_1, s,c,output])
    
    
    ##STEP - 3: DECODER CONTINUATION
    print('Decoder for inference is ready')
    '''Continuer'''
    
    #y_ele = expand_dim(out_input)
    y_ele = y_embedder(out_input)
    print(K.int_shape(y_ele))
    s_repeat = repeator(dec_state)
    s_repeat = densor_1_2(s_repeat)
    s_total = adding_layer([atten_enc,s_repeat])
    s_total = activation_act_0(s_total)
    s_total = densor_att_2(s_total)
    s_total = squeeze_dim(s_total)
    con_weights = attention_act(s_total)
    con_weights = expand_dim_last(con_weights)
    context = dotor(inputs=[con_weights, enc_output])
    context = concatenator([context, y_ele])
    s, _, c = post_layer_lstm(inputs=context, initial_state = [dec_state, dec_c_state])
    output = decode_densor(s)
    print(K.int_shape(output))
    model_dec_step = Model(inputs=[enc_output,atten_enc,dec_state,dec_c_state,out_input], outputs=[s, c, output])
    
    return model, model_enc_inf, model_dec_step
        
		
# custome function for loss and accuracy were to exclude the evaluation of padding while training
def custom_loss(y_true, y_pred):
    mask = K.cast(K.equal(y_true, 0), K.floatx())
    mask = 1 - K.cast(mask, K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.expand_dims(loss,axis=-1)
    loss = loss*mask
    
    #loss = K.sum(loss)/K.sum(mask)
    return loss
	
def custom_accuracy(y_true, y_pred):
    y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    y_pred = K.expand_dims(y_pred,axis=-1)
    mask = K.cast(K.equal(y_true,0),K.floatx())
    mask = 1- K.cast(mask, K.floatx())
    accu = K.equal(y_true,y_pred)
    accu = K.cast(K.equal(y_true,y_pred), K.floatx())
    accu = accu*mask
    return accu