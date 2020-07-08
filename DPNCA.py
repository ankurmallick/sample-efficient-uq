import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from utils import *
from computations import get_embedders, grad_update, get_nca_probs, get_rffweights, get_rff, get_accuracy

# tf.keras.backend.set_floatx('float64')

with open('hyperparams.txt','r') as f:
    hp_list = f.readlines()
    num_epochs = int(hp_list[0][:-1])
    lr = float(hp_list[1])
num_blocks = 10
num_samples = 10
stddev = np.sqrt(2.0) #Scaling factor
log_flag = True

def val_step(X_val,y_val,X,y,embedders_list,latent_weights):
    lab_embeddings = [embedder(X) for embedder in embedders_list]
    lab_embeddings_rff = get_rff(lab_embeddings,latent_weights) 
    val_embeddings = [embedder(X_val) for embedder in embedders_list]
    val_embeddings_rff = get_rff(val_embeddings,latent_weights) 
    val_logits = tf.nn.relu(tf.matmul(val_embeddings_rff,lab_embeddings_rff,transpose_b=True)).numpy()
    val_probs = get_nca_probs(val_logits,y)
    val_acc_check = get_accuracy(val_probs,y_val).numpy()
    return val_acc_check

def train(X,y,M,expt_type):
    #Trains the model on data X and labels y
    if expt_type == 'MNIST':
        latent_dim = 10
    if expt_type == 'COVID':
        latent_dim = 2
    latent_weights = tf.constant(get_rffweights(latent_dim,num_blocks,stddev),dtype=tf.float32)
    # val_acc = 0.0
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, name='Nadam')
    embedders_list = get_embedders(num_samples,latent_dim,expt_type)
    for epoch in range(num_epochs):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            embeddings = [embedder(X) for embedder in embedders_list]
            embeddings_rff = get_rff(embeddings,latent_weights) 
            #Compute Loss
            Kmat = tf.nn.relu(tf.matmul(embeddings_rff,embeddings_rff,transpose_b=True))*M
            Kmat /= tf.reshape(tf.reduce_sum(Kmat,axis=1),[-1,1])
            prob_mat = tf.matmul(Kmat,y) #Prob of x_i picking a point in class j
            prob_vect = tf.reduce_sum(prob_mat*y,axis=1)
            if log_flag == True:
                loss = 0-tf.reduce_mean(tf.math.log(prob_vect))
            else:
                loss = 0-tf.reduce_mean(prob_vect)
        #Get gradients
        params_list = [embedder.variables for embedder in embedders_list]
        grads_list = [tape.gradient(loss,embedder.variables) for embedder in embedders_list]
        if num_samples>1:
        #Update gradients
            grads_list = grad_update(grads_list,params_list,num_samples)
        #Apply gradients
        for (grads,params) in zip(grads_list,params_list):
            optimizer.apply_gradients(zip(grads,params))
        # if epoch%10 == 0 or epoch == num_epochs-1:
        #     val_acc_check = val_step(X_val,y_val,X,y,embedders_list,latent_weights)
        #     print ([epoch, val_acc_check])
        #     if val_acc_check >= val_acc:
        #         val_acc = val_acc_check
        #         best_embedders = [embedder.copy(X_val) for embedder in embedders_list]
        #     else:
        #         continue #Don't save if performance worsens
    best_embedders = [embedder.copy(X) for embedder in embedders_list]
    return best_embedders, latent_weights

def DPNCA_main(X_lab,y_lab,test_points,expt_type):
    # num_lab = X_lab.shape[0]
    # num_val = int(0.2*num_lab)
    # X_val, y_val, X, y = data_splitter(X_lab,y_lab, num_val)
    M = np.ones((X_lab.shape[0],X_lab.shape[0]))
    np.fill_diagonal(M,0)
    M = tf.constant(M,dtype=tf.float32) 
    embedders_list, latent_weights = train(X_lab,y_lab,M,expt_type)  
    lab_embeddings = [embedder(X_lab) for embedder in embedders_list]
    lab_embeddings_rff = get_rff(lab_embeddings,latent_weights) 
    #Checking accuracy on test data
    test_probs = {}
    for key in test_points:
        X_test = test_points[key]
        test_embeddings = [embedder(X_test) for embedder in embedders_list]
        test_embeddings_rff = get_rff(test_embeddings,latent_weights) 
        test_logits = tf.nn.relu(tf.matmul(test_embeddings_rff,lab_embeddings_rff,transpose_b=True)).numpy()
        test_probs[key] = get_nca_probs(test_logits,y_lab)
    return test_probs
