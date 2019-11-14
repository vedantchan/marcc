import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
import pandas as pd
import pickle
import os
plt.rcParams.update({'font.size': 18})
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import inspect

from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adamax
from astroNN.nn import layers as annlayers
from sklearn.preprocessing import MinMaxScaler

class BaseNN:
	'''

	Base class for a simple neural network.

	'''

	def __init__(self, n_input = 14, n_output = 2, n_hidden = 2, neurons = 10, activation = 'relu', output_activation = 'linear', regularization = 0, loss = 'mse', bayesian = False, dropout = 0.1,\
		input_bounds = [[0,1]], output_bounds = [[0,1]], input_scale = None, output_scale = None):

		self.is_initialized = False
		self.n_input = n_input
		self.n_output = n_output
		self.n_hidden = n_hidden
		self.activation = activation
		self.output_activation = output_activation
		self.reg = regularization
		self.neurons = neurons
		self.loss = loss
		self.dropout = dropout
		self.bayesian = bayesian
		self.scaler_isfit = False
		
		if input_scale == 'balmer':
			self.input_bounds = np.array([[0.00000000e+00, 5.36200000e+03],[4.89297491e-10, 2.15776372e+01],[6.55784433e+03, 6.57034846e+03],[4.39400710e+00, 6.96432697e+01],[2.09438209e-03, 4.98649086e+01],\
			[8.20121332e+00, 9.97301468e+01],[1.03210444e-01,5.33331626e-01],[1.05808806e-08,2.38123114e+01],[4.85934060e+03,4.86780698e+03],[4.65600272e+00,7.85555165e+01],[1.58654118e-03,4.91400753e+01],\
			[6.63800862e+00, 9.82804576e+01],[1.39077349e-01, 5.87369186e-01],[1.41997680e-07, 3.41095035e+01],[4.33767599e+03,4.34652748e+03],[3.66673017e+00,4.88370810e+01],[3.28315740e-04,2.69317816e+01],\
			[6.11958791e+00, 8.07541532e+01],[7.43264533e-02, 6.14472692e-01]])
			self.scaler_isfit = True
		
		else:
			self.input_bounds = np.asarray(input_bounds)

		if output_scale == 'labels':
			self.output_bounds = np.asarray([[5000, 80000], [6.5, 9.5]])
			self.scaler_isfit = True
		else:
			self.output_bounds = np.asarray(output_bounds)

		self.input_scaler = MinMaxScaler()
		self.output_scaler = MinMaxScaler()
		self.input_scaler.fit(self.input_bounds.T)
		self.output_scaler.fit(self.output_bounds.T)

	def nn(self):

		x = Input(shape=(self.n_input, ))
		y = Dense(self.neurons, activation = self.activation, kernel_regularizer = l2(self.reg))(x)
		for ii in range(self.n_hidden - 1):
			if self.bayesian:
				y = annlayers.MCDropout(self.dropout)(y)
			y = Dense(self.neurons, activation = self.activation, kernel_regularizer = l2(self.reg))(y)
		if self.bayesian:
			y = annlayers.MCDropout(self.dropout)(y)
		out = Dense(self.n_output, activation = self.output_activation)(y)

		network = Model(inputs = x, outputs = out)
		network.compile(optimizer = Adamax(), loss = self.loss)
		return network

	def train(self, x_data, y_data, model = 'default', n_epochs = 100, batchsize = 64, verbose = 0):
		
		if not self.scaler_isfit:
			print('Warning! Assuming data is scaled. If not, use fit_scaler(X,Y), then train.')

		if model == 'default' and self.is_initialized == False:
			self.model = self.nn()
			self.is_initialized = True
		elif model == 'default' and self.is_initialized == True:
			model = self.model
		x_data = self.input_scaler.transform(x_data)
		y_data = self.output_scaler.transform(y_data)
		h = self.model.fit(x_data, y_data, epochs = n_epochs, verbose = verbose, batch_size = batchsize)
		return h

	def eval(self, x_data, model = 'default', n_bootstrap = 25):
		if model == 'default':
			try:
				model = self.model
			except:
				print('model not trained! use train() or explicitly pass a model to eval()')
				raise
		x_data = self.input_scaler.transform(x_data)
		
		if self.bayesian:
			predictions = np.asarray([self.output_scaler.inverse_transform(self.model.predict(x_data)) for i in range(n_bootstrap)])
			means = np.mean(predictions,0)
			stds = np.std(predictions,0)
			print(means.shape)
			results = np.empty((means.shape[0], means.shape[1] + stds.shape[1]), dtype = means.dtype)
			print(results.shape)
			results[:,0::2] = means
			results[:,1::2] = stds
			return results

		elif not self.bayesian:
			return self.output_scaler.inverse_transform(self.model.predict(x_data))

	def fit_scaler(self, x_data, y_data):
		self.input_bounds = np.asarray([np.min(x_data,0),np.max(x_data,0)]).T
		self.output_bounds = np.asarray([np.min(y_data,0),np.max(y_data,0)]).T

		self.input_scaler.fit(self.input_bounds.T)
		self.output_scaler.fit(self.output_bounds.T)
		print('new bounds established!')
		self.scaler_isfit = True
		return None

	def get_params(self):
		print(inspect.signature(self.__init__))
		return None