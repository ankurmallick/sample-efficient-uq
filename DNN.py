import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from utils import *
from computations import get_embedders,get_accuracy
import time

with open('hyperparams.txt','r') as f:
    hp_list = f.readlines()
    num_epochs = int(hp_list[0][:-1])
    lr = float(hp_list[1])
num_samples = 1
batch_size = 20

def train_step(X,y,embedder,optimizer):
    with tf.GradientTape(persistent=True) as tape:
        cce = tf.losses.CategoricalCrossentropy(from_logits=True )
        # Forward pass
        logits = embedder(X)
        #Compute Loss
        loss = cce(y, logits)
    #Get gradients
    grads = tape.gradient(loss,embedder.variables)
    #Apply gradients
    optimizer.apply_gradients(zip(grads,embedder.variables))
    return embedder, optimizer

# def val_step(X_val,y_val,embedder):
#     val_logits = embedder(X_val)
#     val_acc_check = get_accuracy(val_logits,y_val).numpy()
#     return val_acc_check

def train(X,y,expt_type):
    #Trains the model on data X and labels y
    if expt_type == 'MNIST':
        latent_dim = 10
    if expt_type == 'COVID':
        latent_dim = 2
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.shuffle(1000).batch(batch_size)
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, name='Nadam')
    embedder = get_embedders(num_samples,latent_dim,expt_type)[0]
    for epoch in range(num_epochs):
        #Training
        for (batch, (points, targets)) in enumerate(dataset):
            embedder, optimizer = train_step(points,targets,embedder,optimizer)
        #Validation
        # if epoch%10 == 0 or epoch == num_epochs-1:
        #     val_acc_check = val_step(X_val,y_val,embedder)
        #     print ([epoch, val_acc_check])
        #     if val_acc_check >= val_acc:
        #         val_acc = val_acc_check
        #         best_embedder = embedder.copy(X_val) #Update best model
        #     else:
        #         continue #Don't save if performance worsens
    best_embedder = embedder.copy(X)
    return best_embedder

def DNN_main(X_lab,y_lab,test_points,expt_type):
    # num_lab = X_lab.shape[0]
    # num_val = int(0.2*num_lab)
    # X_val, y_val, X, y = data_splitter(X_lab,y_lab, num_val)
    embedder = train(X_lab,y_lab,expt_type)  
    #Checking accuracy on test data
    test_probs = {}
    for key in test_points:
        X_test = test_points[key]
        test_logits = embedder(X_test)
        test_probs[key] = tf.nn.softmax(test_logits,axis=1).numpy()
    return test_probs
