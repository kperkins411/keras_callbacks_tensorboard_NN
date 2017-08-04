'''
Test for keras callbacks, no normalization, or train/test split
just a test
'''
# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
import numpy
import logging

logging.basicConfig(
    # filename = 'parse_data.log', #comment this line out if you want data in the console
    format = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s",
    level = logging.DEBUG
)


def makeModel(weightfile=None):
    # create model
    model = Sequential()
    # try glorot_normal for initilizer
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # load weights if given
    if weightfile != None:
        model.load_weights(weightfile)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('acc'))


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
'''
   each row has 9 columns, means are

   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

   Missing Attribute Values: Yes

   Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268

   Brief statistical analysis:
    Attribute number:    Mean:   Standard Deviation:
    1.                     3.8     3.4
    2.                   120.9    32.0
    3.                    69.1    19.4
    4.                    20.5    16.0
    5.                    79.8   115.2
    6.                    32.0     7.9
    7.                     0.5     0.3
    8.                    33.2    11.8

   dataset is unbalanced and needs to be normalized
'''

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#ceate a model
model = makeModel()

#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"  #creates a file with name of epoch and accuracy as part of name
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
losshistory = LossHistory()
tensorboard = TensorBoard()

#add all the callbacks to a list
callbacks_list = [checkpoint,losshistory, tensorboard]

# Fit the model, with callbacks
model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=10, callbacks=callbacks_list, verbose=0)

#lets see what sort of infor was saved
logging.debug(losshistory.losses)

# lets reload the best weights and fit the model
model =  makeModel(filepath)

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
logging.debug("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
