# keras_callbacks_tensorboard_NN
example of callbacks to checkpoint a model, there are 3 callbacks<br>

* LossHistory - custom, derived from Callback
* ModelCheckpoint - builtin
* tensorboard - built in


tensorboard is interesting, it will create a log file (in /logs folder).  Visualize the data by this command:<br>
tensorboard --logdir=path/to/log-directory
<br>and then browser view the page and port it indicates (ex. http://keith-XPS-15-9550:6006)


## get model data from NIH by running following command
wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data
