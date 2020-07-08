import numpy as np
import tensorflow as tf
from utils import *
from computations import get_embedders,get_accuracy

with open('hyperparams.txt','r') as f:
    hp_list = f.readlines()
    num_epochs = int(hp_list[0][:-1])
    lr = float(hp_list[1])
num_samples = 10
batch_size = 20

def train_step(X,y,embedders_list,optimizer):
    with tf.GradientTape(persistent=True) as tape:
        cce = tf.losses.CategoricalCrossentropy(from_logits=True )
        # Forward pass
        losses_list = []
        for embedder in embedders_list:
            logits = embedder(X)
            losses_list.append(cce(y, logits))
        # loss = tf.reduce_mean(tf.stack(losses_list)) 
    #embedder_update
    params_list = [embedder.trainable_variables for embedder in embedders_list]
    grads_list = [tape.gradient(loss,embedder.variables) for (loss,embedder) in zip(losses_list,embedders_list)]
    #Apply gradients
    for (grads,params) in zip(grads_list,params_list):
        optimizer.apply_gradients(zip(grads,params))
    return embedders_list, optimizer

def val_step(X_val,y_val,embedders_list):
    probs_list = []
    for embedder in embedders_list:
        logits = embedder(X_val)
        probs = tf.nn.softmax(logits)
        probs_list.append(probs)
    val_probs = tf.reduce_mean(tf.stack(probs_list),axis=0)
    val_acc_check = get_accuracy(val_probs,y_val).numpy()
    return val_acc_check

def train(X,y,expt_type):
    #Trains the model on data X and labels y
    if expt_type == 'MNIST':
        latent_dim = 10
    if expt_type == 'COVID':
        latent_dim = 2
    # val_acc = 0.0
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.shuffle(1000).batch(batch_size)
    optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, name='Nadam')
    embedders_list = get_embedders(num_samples,latent_dim,expt_type)
    for epoch in range(num_epochs):
        #Training
        for (batch, (points, targets)) in enumerate(dataset):
            embedders_list, optimizer = train_step(points,targets,embedders_list,optimizer)
        #Validation
        # if epoch%10 == 0 or epoch == num_epochs-1:
        #     val_acc_check = val_step(X_val,y_val,embedders_list)
        #     print ([epoch, val_acc_check])
        #     if val_acc_check > val_acc:
        #         val_acc = val_acc_check
        #         best_embedders = [embedder.copy(X_val) for embedder in embedders_list]#Update best model
        #     else:
        #         continue #Don't save if performance worsens
    best_embedders = [embedder.copy(X) for embedder in embedders_list]
    return best_embedders

def Ensemble_main(X_lab,y_lab,test_points,expt_type):
    # num_lab = X_lab.shape[0]
    # num_val = int(0.2*num_lab)
    # X_val, y_val, X, y = data_splitter(X_lab,y_lab, num_val)
    embedders_list = train(X_lab,y_lab,expt_type)  
    #Checking accuracy on test data
    test_probs = {}
    for key in test_points:
        X_test = test_points[key]
        probs_list = []
        for embedder in embedders_list:
            logits = embedder(X_test)
            probs = tf.nn.softmax(logits)
            probs_list.append(probs)
        test_probs[key] = tf.reduce_mean(tf.stack(probs_list),axis=0).numpy()
    return test_probs
