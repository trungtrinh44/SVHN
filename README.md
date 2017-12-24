# SVHN
- This repository contains 3 models for Street View House Number classification.
- The first model is Histogram of Oriented Gradient combined with Neural Network 
which is contained in the folder HOG_ANN. The script ```train_hog_ann.py``` is used to 
train this model on training set and ```eval_hog_ann.py``` is used to evaluate this model
 on test set.
- The second model is Convolutional Neural Network which is contained in the folder CNN. The script ```train_cnn.py``` is used
 to train this model on training set and ```eval_cnn.py``` is used to evaluate this model on test set.
- The third model is Histogram of Oriented Gradient combined with Support Vector Machine. The script ```train_hog_svm.py``` is used
 to train this model on training set and ```eval_hog_svm``` is used to evaluate this model on test set.
- To run these script first download the training and testing data from [SVHN](http://ufldl.stanford.edu/housenumbers/).
- To train the CNN model run ```python train_cnn.py '--data_dir [path to data] --log_dir [path for log files]'```
- To train the HOG_ANN model run ```python train_hog_ann.py '--data_dir [path to data] --log_dir [path for log files]'```
- To evaluate the CNN model run ```python eval_cnn.py --model_dir [path to trained model] --data_dir [path to data]```
- To evaluate the HOG_ANN model run ```python eval_hog_ann.py --model_dir [path to trained model] --data_dir [path to data]```
- To train the HOG_SVM model run ```python train_hog_svm.py --data_dir [path_to_data]```
- To evaluate the HOG_SVM model run ```python eval_hog_svm.py --model_dir [path to trained model] --data_dir [path to data]```