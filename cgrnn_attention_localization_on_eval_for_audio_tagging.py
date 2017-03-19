import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config_2ch_raw_mbk_ipld_eva as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_2ch_raw_ipd_ild_easy as pp_data
#from prepare_data import load_data

import keras
from keras import backend as K

from keras.datasets import mnist, cifar10
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Activation,Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.utils import np_utils
from keras.layers import Merge, Input, merge
from keras.constraints import nonneg
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
import h5py

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 6, t_delay, feadim, 1) )

def reshapeX1( X ):
    N = len(X)
    return X.reshape( (N, t_delay, 1, feadim, 1) )

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim) )

def get_R(X):
    a,b=X.values()
    return Keras.dot(a,b)

def outfunc(vects):
    x,y=vects
    #y=K.sum( y, axis=1 )
    y = K.clip( y, 1.0e-9, 1 )     # clip to avoid numerical underflow
    #z=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(y)
    y = K.sum(y, axis=1)
    #z = RepeatVector(249)(z)
    #z=Permute((2,1))(z)
    #return K.sum( x / z, axis=1 )
    return K.sum( x, axis=1 ) / y

def outfunc_v2(vects):
    x,y=vects
    #y=K.sum( y, axis=1 )
    x=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(x)
    y=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(y)
    return x/y ###maybe the same with the outfunc, because the the denominator is not related to axis=1



feadim=40
t_delay=249 # the len of Utterance is 249

# hyper-params
fe_fd_left = cfg.dev_fe_mel_fd_left
fe_fd_right = cfg.dev_fe_mel_fd_right
fe_fd_mean = cfg.dev_fe_mel_fd_mean
fe_fd_diff = cfg.dev_fe_mel_fd_diff
fe_fd_ipd = cfg.dev_fe_mel_fd_ipd
fe_fd_ild = cfg.dev_fe_mel_fd_ild

#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = t_delay        # concatenate frames
hop = t_delay            # step_len
n_hid = 1000
n_out = len( cfg.labels )
print n_out
fold = 9         # can be 0, 1, 2, 3, 4

# prepare data
tr_X1, tr_X2, tr_y, te_X1, te_X2, te_y = pp_data.GetAllData_separate( fe_fd_right, fe_fd_left, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, agg_num, hop, fold )
#[batch_num, inmap, n_time, n_freq] = tr_X1.shape
print tr_X1.shape
print tr_X2.shape
#sys.exit()
tr_X1, te_X1 = reshapeX1(tr_X1), reshapeX1(te_X1)
##tr_X2, te_X2 = reshapeX1(tr_X2), reshapeX1(te_X2)
    
print tr_X1.shape, tr_X2.shape, tr_y.shape
print te_X1.shape, te_X2.shape, te_y.shape

###build model by keras
kernel_size=(30,1)
pool_size=(40-30+1,1)
#pool_size=(3,1)


input_audio=Input(shape=(t_delay, 1, feadim, 1))

cnn=TimeDistributed(Convolution2D(128, 30, 1, border_mode='valid', bias=True, activation='relu'))(input_audio)
mp=TimeDistributed(MaxPooling2D(pool_size=pool_size))(cnn)

flat=TimeDistributed(Flatten())(mp)
#model1.add(Reshape((t_delay, feadim),input_shape=(t_delay,feadim)))

###detection factor for each tag (7 meaningful tags + 1 silence tag = 8 tags)
#det =TimeDistributed(Dense(500,activation='relu'))(flat)
det =TimeDistributed(Dense(8,activation='softmax'))(flat) # The posterior sum of each tag is 1.0, now the dims of det are 33 frs * 8 tags

lstm=Bidirectional(GRU(output_dim=128,return_sequences=True,dropout_W=0.0, dropout_U=0.0))(flat)
#print model.output_shape
lstm=Bidirectional(GRU(output_dim=128,return_sequences=True,dropout_W=0.0, dropout_U=0.0))(lstm)
lstm=Bidirectional(GRU(output_dim=128,return_sequences=True,dropout_W=0.0, dropout_U=0.0))(lstm)

#att = Flatten()(lstm)
#att = merge([lstm,flat], mode='concat')
#att = Dense(500,activation='relu')(att)
#att = TimeDistributed(Dense(500,activation='relu'))(flat)
att = TimeDistributed(Dense(1,activation='sigmoid'))(flat)
att = Flatten()(att)
att1 = RepeatVector(256)(att)
att1=Permute((2,1))(att1)
#att2 = RepeatVector(8)(att)
#att2=Permute((2,1))(att2)

merge1 = merge([lstm, att1], mode='mul') #Merge is for model, while merge is for tensor.  # concat_axis=2 # two matrix have same dims, 'mul' should be dot2dot mult
#det2 = merge([att2, det], mode='mul')
#merge2=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(merge2) # direct K.sum did not include input_shape info which is needed by next layer

#merge1=Flatten()(merge1)
dnn=TimeDistributed(Dense(500,activation='relu'))(merge1)
dnn=TimeDistributed(Dropout(0.2))(dnn)
dnn=TimeDistributed(Dense(n_out,activation='sigmoid'))(dnn)
dnnout = merge([dnn, det], mode='mul') 
#dnnout=merge([dnn, att2], mode='mul')
#dnnout=Lambda(lambda x: K.mean(x, axis=1),output_shape=(8,))([dnnout])
dnnout=Lambda(outfunc,output_shape=(8,))([dnnout,det])

#dnnout=merge([merge2, dnn], mode='mul')
#dnnout=Lambda(lambda x: K.sum(x, axis=1),output_shape=(8,))(dnnout)
##dnnout=Flatten()(dnnout)
#dnnout=Dense(n_out,activation='sigmoid')(dnnout)

allmodel=Model(input_audio, dnnout)
allmodel.summary()

#adam=keras.optimizers.Adam(lr=1e-4)
allmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dump_fd=cfg.scrap_fd+'/Md/Attention_detection_clip_Mult3V2_utt_keras_overlap50_eva816_1CNN128onMBK40_noILD_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')    

#model1.fit([tr_X1,tr_X2], tr_y, batch_size=100, nb_epoch=101,
#              verbose=1, validation_data=([te_X1,te_X2], te_y), callbacks=[eachmodel]) #, callbacks=[best_model])

allmodel.fit(tr_X1, tr_y, batch_size=100, nb_epoch=101,
              verbose=1, validation_data=(te_X1, te_y), callbacks=[eachmodel]) #, callbacks=[best_model])

