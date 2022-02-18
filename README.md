# sample-efficient-uq
Code for our paper 'Probabilistic Neighbourhood Component Analysis: Sample Efficient Uncertainty Estimation in Deep Learning' (https://arxiv.org/abs/2007.10800) which:

a. Appeared at the ICML 2020 Workshop on Uncertainty in Deep Learning (UDL) (https://sites.google.com/view/udlworkshop2020/home). 

b. Featured in Andrew Ng’s weekly digest on deep learning research highlights (https://dlai.carney.co/the-batch/issue-51/). 

Run 'run_expts_MNIST.py' to obtain accuracy and confidence values (as defined in the paper) for different approaches - DNN, BNN, Ensemble, PNCA (Ours) with MNIST as train-data and three test datasets:

1. Org: Original MNIST test data. 
2. Rot: MNIST digits rotated by 60 degrees. 
3. OOD: Images of letters (not digits). 

Results are saved for different sizes of training datasets (uncertainty quantification is harder for smaller dataset sizes)
