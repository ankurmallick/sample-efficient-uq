import pickle
import sys
import numpy as np
import tensorflow as tf
from utils import *
from DNN import DNN_main
from BNN import BNN_main
from DNCA import DNCA_main
from DPNCA import DPNCA_main
from Ensemble import Ensemble_main
from computations import get_acc_vect

#Experiment settings
num_classes = 10
num_trials = 10
num_lab = int(sys.argv[1])
model_funcs = {'DNN':DNN_main,'BNN':BNN_main,'DNCA':DNCA_main,'DPNCA':DPNCA_main,'Ensemble':Ensemble_main}
num_models = len(model_funcs)
# model_funcs = {'DNN':DNN_main,'DNCA':DNCA_main,'DPNCA':DPNCA_main}

#Import Data
test_keys = ['Org','Rot','OOD']
test_points = {}
test_labels = {}
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = process(X_train,y_train,num_classes)
test_points['Org'],test_labels['Org'] = process(X_test,y_test,num_classes)
test_points['Rot'],test_labels['Rot'] = process(rotate(X_test),y_test,num_classes)
test_points['OOD'],test_labels['OOD'] = process(np.load('notMNIST/X.npy'),np.load('notMNIST/y.npy').astype(int),num_classes)
acc = {}
conf = {}
for key in test_keys:
    num_test = test_points[key].shape[0]
    acc[key] = [np.zeros((num_trials,num_test)) for _ in range(num_models)]
    conf[key] = [np.zeros((num_trials,num_test)) for _ in range(num_models)]

#Running Experiments
for trial in range(num_trials):
    print ('Trial = '+str(trial))
    X_lab, y_lab, X_train, y_train = data_splitter(X_train, y_train, num_lab)
    for (ctr,model) in enumerate(sorted(model_funcs.keys())):
        print (model)
        test_probs = model_funcs[model](X_lab,y_lab,test_points,'MNIST')
        for key in test_keys:
            acc[key][ctr][trial] = get_acc_vect(test_probs[key],test_labels[key])
            conf[key][ctr][trial] = np.amax(test_probs[key],axis=1)
            print (key + ' Accuracy = '+str(np.mean(acc[key][ctr][trial])))
            print (key + ' Confidence = '+str(np.mean(conf[key][ctr][trial])))

#Saving Results
with open('Results/MNIST/acc_'+str(num_lab)+'.pkl','wb') as f:
    pickle.dump(acc,f)
with open('Results/MNIST/conf_'+str(num_lab)+'.pkl','wb') as f:
    pickle.dump(conf,f)
