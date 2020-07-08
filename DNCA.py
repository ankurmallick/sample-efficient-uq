import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from utils import *
from computations import scaled_sqdist, get_nca_probs, get_accuracy, get_embedders

# tf.keras.backend.set_floatx('float64')

with open('hyperparams.txt','r') as f:
    hp_list = f.readlines()
    num_epochs = int(hp_list[0][:-1])
    lr = float(hp_list[1])
num_samples = 1
log_flag = True

def val_step(X_val,y_val,X,y,embedder):
    lab_embeddings = embedder(X)
    val_embeddings = embedder(X_val)
    val_logits = tf.exp(-0.5*scaled_sqdist(val_embeddings,lab_embeddings)).numpy()
    val_probs = get_nca_probs(val_logits,y)
    val_acc_check = get_accuracy(val_probs,y_val).numpy()
    return val_acc_check

def train(X,y,M,expt_type):
    #Trains the model on data X and labels y
    if expt_type == 'MNIST':
        latent_dim = 10
    if expt_type == 'COVID':
        latent_dim = 2
    # val_acc = 0.0
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, name='Nadam')
    embedder = get_embedders(num_samples,latent_dim,expt_type)[0]
    for epoch in range(num_epochs):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            embeddings = embedder(X)
            #Compute Loss
            Kmat = tf.exp(-0.5*scaled_sqdist(embeddings))*M
            Kmat /= tf.reshape(tf.reduce_sum(Kmat,axis=1),[-1,1])
            prob_mat = tf.matmul(Kmat,y) #Prob of x_i picking a point in class j
            prob_vect = tf.reduce_sum(prob_mat*y,axis=1)
            if log_flag == True:
                loss = 0-tf.reduce_mean(tf.math.log(prob_vect))
            else:
                loss = 0-tf.reduce_mean(prob_vect)
        #Get gradients
        params = embedder.variables
        grads = tape.gradient(loss,embedder.variables)
        optimizer.apply_gradients(zip(grads,params))
        # if epoch%10 == 0 or epoch == num_epochs-1:
        #     val_acc_check = val_step(X_val,y_val,X,y,embedder)
        #     print ([epoch, val_acc_check])
        #     if val_acc_check >= val_acc:
        #         val_acc = val_acc_check
        #         best_embedder = embedder.copy(X_val) #Update best model
        #     else:
        #         continue #Don't save if performance worsens
    best_embedder = embedder.copy(X)
    return best_embedder

def DNCA_main(X_lab,y_lab,test_points,expt_type):
    # num_lab = X_lab.shape[0]
    # num_val = int(0.2*num_lab)
    # X_val, y_val, X, y = data_splitter(X_lab,y_lab, num_val)
    M = np.ones((X_lab.shape[0],X_lab.shape[0]))
    np.fill_diagonal(M,0)
    M = tf.constant(M,dtype=tf.float32) 
    embedder = train(X_lab,y_lab,M,expt_type)  
    lab_embeddings = embedder(X_lab)
    #Checking accuracy on test data
    test_probs = {}
    for key in test_points:
        X_test = test_points[key]
        test_embeddings = embedder(X_test)
        test_logits = tf.exp(-0.5*scaled_sqdist(test_embeddings,lab_embeddings)).numpy()
        test_probs[key] = get_nca_probs(test_logits,y_lab)
    return test_probs
