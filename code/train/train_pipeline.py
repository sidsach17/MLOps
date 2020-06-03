
import pandas as pd
import glob
import numpy as np
import cv2
#from tqdm import tqdm
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization,Activation,LeakyReLU
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.applications.densenet import preprocess_input
import tensorflow as tf
from keras.utils import multi_gpu_model, Sequence
import os
import sys
import copy
from random import shuffle
from keras import backend as K
import multiprocessing as mp
from azureml.core import Run
import argparse
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tfback
from tensorflow.keras.callbacks import EarlyStopping
import json
import pickle

model_path = 'outputs/'

################### Azure ML Service Starts #############################
"""parser = argparse.ArgumentParser()
parser.add_argument('--w_minus', dest='w_minus', required=True)
parser.add_argument('--n_by_p', dest='n_by_p', required=True)
parser.add_argument('--x', dest='x', required=True)
parser.add_argument('--y', dest='y', required=True)
parser.add_argument('--training_generator', dest='training_generator', required=True)
parser.add_argument('--validation_generator', dest='validation_generator', required=True)"""



parser = argparse.ArgumentParser()
parser.add_argument('--step1_param', dest='step1_param', required=True)
parser.add_argument('--epochs', type=int, dest='epochs', help='No. of epochs', default=2)
parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size', default =32)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', help='learning_rate', default =0.001)
args = parser.parse_args()

step1_param = args.step1_param
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate


with open(os.path.join(step1_param, 'step1param.json')) as json_file:
    data = json.load(json_file)
    w_minus = data['w_minus']
    n_by_p = data['n_by_p']
    x = data['x']
    y = data['y']

with open(os.path.join(step1_param, 'training_generator.pkl'), 'rb') as input_train:
    training_generator = pickle.load(input_train)    

    
with open(os.path.join(step1_param, 'validation_generator.pkl'), 'rb') as input_valid:
    validation_generator = pickle.load(input_valid)    

print(type(training_generator))
print(type(validation_generator))



print("test")
training_generator = (n for n in training_generator)
validation_generator = (n for n in validation_generator)

"""w_minus = args.w_minus
n_by_p = args.n_by_p
x = args.x
y = args.y
training_generator = args.training_generator
validation_generator = args.validation_generator"""


"""print(w_minus)
print(type(w_minus))
print(n_by_p)
print(type(n_by_p))
print(x)
print(type(x))
print(y)
print(type(y))
print(training_generator)
print(type(training_generator))
print(validation_generator)
print(type(validation_generator))"""

# learning and decay can also be passed in the same way
################## Azure ML Service Ends ################################

# learning and decay can also be passed in the same way


"""parser = argparse.ArgumentParser()
parser.add_argument('--pd', dest='pd', required=True)
args = parser.parse_args()

with open(args.pd, 'r') as f:
    f.read('w_minus')
    f.read('n_by_p')
    f.read('x')
    f.read('y')
    f.read('training_generator')
    f.read('validation_generator')"""
    
##############################Building Model############################################################
from sklearn.metrics import accuracy_score,roc_auc_score
from keras.optimizers import Adam

print('Building Model...')
#with tf.device('/cpu:0'):
model=Sequential()
model.add(DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3)))#,dropout_rate=0.5))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(14,activation='sigmoid'))# adds last layer to have 14 neurons [[14 ,kernel_regularizer=regularizers.l2(0.01))]]
model.summary()
parallel_model = model
#parallel_model = multi_gpu_model(model, gpus=2) # model has been built i.e the structure of CNN model has been built



def custom_loss(targets, output):
    _epsilon = tf.convert_to_tensor(K.epsilon(),output.dtype.base_dtype)#, output.dtype.base_dtype
    output = tf.clip_by_value(tf.cast(output,tf.float32), _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    weight=tf.convert_to_tensor(w_minus,dtype='float32')
    return K.mean(weight*tf.nn.weighted_cross_entropy_with_logits(logits=output, labels=targets, pos_weight=np.array(n_by_p), name=None),axis=-1)
    
"""def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        Wrapper for turning tensorflow metrics into keras metrics 
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
auc_roc = as_keras_metric(tf.metrics.auc)"""

auc_roc = tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
    dtype=None, thresholds=None, multi_label=True, label_weights=None)

optimizer=Adam(learning_rate,decay=1e-5)#0.001

# tell the model what loss function and what optimization method to use
parallel_model.compile(loss=custom_loss,optimizer=optimizer,metrics=[auc_roc])

####################Azure ML Service Specific Code###################################
# start an Azure ML run
run = Run.get_context()
####################Azure ML Service Specific Code###################################



######################################## Model Building Ends here ##########################################

####################################### Model Training & Validation Starts ##############################################
from keras.callbacks import ReduceLROnPlateau
            
class Histories(keras.callbacks.Callback):
    def __init__(self,model):
        self.model_for_saving = model
        self.monitor_op = np.less
        self.best = np.Inf
        
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        print("training started")

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #run.log('Loss',logs['loss'])
        current=logs.get('val_loss')
        ####################Azure ML Service Specific Code###################################
        run.log('train_Loss',logs['loss'])
        run.log('val_Loss',logs['val_loss'])
        run.log('train_auc',logs['auc'])
        run.log('val_auc',logs['val_auc'])
        ####################Azure ML Service Specific Code###################################
        
        #run.log('loss',logs['loss'],'val_Loss',logs['val_loss'])
        #run.log('val_Loss',logs['val_loss'])
        #run.log_row(logs['loss'],logs['val_loss'])
        """plt.plot(run.log('Loss',logs['loss']))
        plt.plot(run.log('val_loss',logs['val_loss']))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(['train','valid'], loc='upper left')"""
        if self.monitor_op(current, self.best):
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                  ' saving model'
                  % (epoch + 1, 'val_loss', self.best,
                     current))
                     
            self.best = current
            self.model_for_saving.save(os.path.join(model_path,'weights.best.dense_generator_callback.hdf5'))
            #run.parent.upload_file(name = os.path.join(model_path,'weights.best.dense_generator_callback.hdf5'), path_or_stream = 'weights.best.dense_generator_callback.hdf5')
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, 'val_loss', self.best))
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

auc=Histories(model=model)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, mode='min',min_lr=.00001,cooldown=1)
monitor = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

def train():
    import keras.backend as k
    history=parallel_model.fit_generator(training_generator,
                        steps_per_epoch=x // batch_size, epochs=epochs,
                       validation_data=validation_generator,
                        validation_steps=y // batch_size,
                         verbose=1,
                        workers=16,
                        use_multiprocessing=True,
                        callbacks=[auc,reduce_lr,monitor]
                       )#
    pd.DataFrame(history.history).to_csv(os.path.join(model_path,'history.csv'),index=False)
    df = pd.read_csv(os.path.join(model_path,'history.csv'))
    
    ####################Azure ML Service Specific Code###################################
    
    fig1 = plt.figure()
    plt.plot(df['loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Training Loss',plot=fig1) 

    df1 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig2 = plt.figure()
    plt.plot(df1['val_loss'])
    plt.title('validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Validation Loss',plot=fig2)
    
    df2 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig3 = plt.figure()
    plt.plot(df2['auc'])
    plt.title('Training AUC')
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Training AUC',plot=fig3)
    
    df3 = pd.read_csv(os.path.join(model_path,'history.csv'))
    fig4 = plt.figure()
    plt.plot(df3['val_auc'])
    plt.title('Validation AUC')
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    run.log_image('Epochs vs Validation AUC',plot=fig4)
    ####################Azure ML Service Specific Code###################################
    model = run.register_model(model_name='pipeline_demo', model_path='outputs/weights.best.dense_generator_callback.hdf5')
    
    

if __name__ == '__main__':
    
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
