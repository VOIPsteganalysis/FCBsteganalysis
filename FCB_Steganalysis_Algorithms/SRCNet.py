#!/usr/bin/env python
#-*-coding:utf-8 -*-

"""
@author: Chen Gong

"""

import numpy as np
import os, pickle, random, datetime
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Bidirectional, LSTM, GlobalAveragePooling1D
from keras.callbacks import TensorBoard

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

# net para
BATCH_SIZE = 32         # batch size
nb_epochs = 30          # number of iteration
FOLD = 5                # = NUM_SAMPLE / number of testing samples

'''
Get the paths of all files in the folder
'''
def get_file_list(folder):
	file_list = []
	for file in os.listdir(folder):
		file_list.append(os.path.join(folder, file))
	return file_list

'''
Read codeword file
'''	
def parse_sample(file_path):
	file = open(file_path, 'r')
	lines = file.readlines()
	sample =[]
	for line in lines:
		line_split = line.strip("\r\n\t").strip().split(" ")
		sample.append(line_split)
	return sample

'''
Save variable in pickle
'''
def save_variable(file_name, variable):
	file_object = open(file_name, "wb")
	pickle.dump(variable, file_object)
	file_object.close()

'''
Main
'''
if __name__ == '__main__':

	SRCNet = [
	{"class" : 1, "folder" : "/the/path/to/Stego"}, # The folder that contains positive data files.
	{"class" : 0, "folder" : "/the/path/to/Cover"}  # The folder that contains negative data files.
]

	model_save_dir = os.path.join(os.getcwd(), 'Trained_models')
	model_name = 'SRCNet'
	model_path = os.path.join(model_save_dir, model_name)
	log_save_dir = os.path.join(os.getcwd(), 'Log')
	log_name = 'SRCNet.txt'
	log_file = os.path.join(log_save_dir, log_name)
	TensorBoard_path = os.path.join(os.getcwd(), 'TensorBoard/SRCNet')
	isExists=os.path.exists(TensorBoard_path)
	if not isExists:
		os.makedirs(TensorBoard_path) 
	isExists=os.path.exists(model_save_dir)
	if not isExists:
		os.makedirs(model_save_dir) 
	isExists=os.path.exists(log_save_dir)
	if not isExists:
		os.makedirs(log_save_dir) 
	
	all_files = [(item, folder["class"]) for folder in SRCNet for item in get_file_list(folder["folder"])]
	random.shuffle(all_files)
	save_variable('all_files.pkl', all_files)
	
	all_samples_x = [(parse_sample(item[0])) for item in all_files]
	all_samples_y = [item[1] for item in all_files]
	
	np_all_samples_x = np.asarray(all_samples_x)
	np_all_samples_y = np.asarray(all_samples_y)
	
	save_variable('np_all_samples_x.pkl', np_all_samples_x)
	save_variable('np_all_samples_y.pkl', np_all_samples_y)
	
	file_num = len(all_files)
	sub_file_num = int(file_num / FOLD)
	
	x_test = np_all_samples_x[0 : sub_file_num]
	y_test = np_all_samples_y[0 : sub_file_num]
	
	x_train = np_all_samples_x[sub_file_num : file_num]
	y_train = np_all_samples_y[sub_file_num : file_num]
	
	print("Building model")
	model = Sequential()
	model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape = (100, 3)))
	model.add(GlobalAveragePooling1D())
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ["accuracy"])

	print("\nTraining\n")
	for i in range(nb_epochs):
		startdate = datetime.datetime.now() 
		print('\nstart date: \n',startdate)
		start = time.time()
		hist = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = 1, validation_data = (x_test, y_test),
						callbacks=[TensorBoard(log_dir=TensorBoard_path)])
		end = time.time() 
		train_time = end-start
		enddate = datetime.datetime.now()
		print('\nend date:  \n',enddate)
		print('\nTraining Time: ' + str(train_time) + '.seconds \n' )
		model.save( model_path + '_%d.h5' % (i + 1))
	
		# save result	
		with open( log_file,'a+') as f:
			f.write(str(hist.history) +'\n')
			f.write('epoch: '+ str(i) +'\n')
			f.write('x_train shape: ' + str(x_train.shape) + '\n')
			f.write('x_test shape: ' + str(x_test.shape)+'\n')
			f.write('training acc: ' + str(hist.history['acc'][-1]) + '\n')
			f.write('training loss: ' + str(hist.history['loss'][-1]) + '\n')
			f.write('val_acc: ' + str(hist.history['val_acc'][-1]) + '\n')
			f.write('val_loss: ' + str(hist.history['val_loss'][-1]) + '\n')
			f.write('train_time: ' + str(train_time) + '\n')
			f.write('---------------------------------------------------------------------------\n')
			f.write('\r\n')