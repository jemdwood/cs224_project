import tensorflow as tf
import numpy as np
import lasagne as las
import theano
import time
import csv
import pandas as pd
import nolearn
from encodings.punycode import digits
from lasagne.nonlinearities import softmax
from lasagne.updates import apply_momentum
from lasagne.updates import nesterov_momentum
#from lasagne.objectives import binary_hinge_loss
from sklearn.preprocessing.label import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from nolearn.lasagne import NeuralNet
from numpy import float64
from lasagne.objectives import binary_hinge_loss


def feature_extractions():
    return {"current population": lambda x: x["cur_pop"],
            "last population": lambda x: x["last_pop"],
     "extra": lambda x: x["extra"],
     "inverse current pop": lambda x: 1.0/x["cur_pop"],
     "inverse square current pop": lambda x: 1.0/(x["cur_pop"]**2),
     "inverse last pop": lambda x: 1.0/x["last_pop"],
     "inverse square last pop": lambda x: 1.0/(x["last_pop"]**2)
    }

def preprocessed_data():
    return "preprocessed_data.csv"

def featurize(i):
    a = np.array(map(lambda fn: fn(i), feature_extractions().values()))
    return a

# res moved to
def read_cur_res(info):
    return str(info["curr_res"])

#place moved from
def read_last_res(info):
    return str(info["last_res"])

def read_uniq_key(info_dict):
    return int(read_cur_res(info_dict) + read_last_res(info_dict))

def supplement(info):
    #add info about city 
    pass
    # TODO
    return info

def normalize_arrays(xs):
    # TODO / might not be needed, investigate
    pass

def write_preprocessed_csv(xs,ys):
    with open(preprocessed_data(), "wb") as csvfile:
        writer = csv.writer(csvfile)
        header = feature_extractions().keys()
        header.append("output")
        writer.writerow(header)
        for r in range(len(xs)):
            to_write = xs[r].tolist()
            to_write.append(ys[r])
            writer.writerow(to_write)
        csvfile.close()

def load_xs_and_ys():
    f = open("CtyxCty_US_Census_BuildingData.txt", "r+")
    ys = []
    xs = []
    infos = []
    a = 0
    for line in f:
        try:
            d = extract_line_info(line)
        except:
            a += 1
            continue
        infos.append(d)
    print "%i samples not processed out of %i (%f)" % (a, len(infos), 1.0*a/len(infos))
    infos.sort(key = read_uniq_key)
    infos = supplement(infos)
    for i in infos:
        ys.append(i["movers"])
        xs.append(featurize(i))
    write_preprocessed_csv(xs, ys)
    return xs, ys

def read_file_building():
    xs, ys = load_xs_and_ys() #actually throw away and just use the written file. Oopsies
    df = pd.read_csv(preprocessed_data())
    X = df.values.copy()
    X, y = X[:, 0:-1].astype(np.float64), X[:, -1].astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler
        
def extract_line_info(line):
    #HARDCODING TO BE REMOVED
    last_res_pop = int(line[259:267])
    cur_res_pop = int(line[79:87])
    curr_res_county_code = line[3:6]
    last_res_county_code = line[9:12] #should be three digits
    movers = int(line[373:380])
    extra = int(line[309:316])
    return {"extra": extra, 'cur_pop': cur_res_pop, "last_pop": last_res_pop, "curr_res": curr_res_county_code, "last_res": last_res_county_code, "movers": movers}

# THIS Needs to be improved to train off of a year at a time and then predict for the next year
def partition_data(xs, ys):
    test_perc = 0.15
    partition = int((1.0-test_perc)*len(xs))
    train = [xs[0:partition], ys[0:partition]]
    test = [xs[partition:], ys[partition:]]
    return train, test

def get_loss_function(scaler):
    #------squared loss
    def loss_function (a,b):
        print "\/" # TODO
        print a-b
        transformed= a#scaler.transform(np.array(a)[0])
        return (transformed-b)**2
    return loss_function

#This code pulled from Lasagne examples: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_net(train, test, y_scaler):
    xs_test, ys_test = test
    xs_train, ys_train = train
    num_features = xs_train.shape[1]
    assert(num_features == len(feature_extractions().keys()))
    loss_function = get_loss_function(y_scaler)
    
    input_var = theano.tensor.dmatrix('inputs')
    target_var = theano.tensor.dvector('targets')
    
    l_in = las.layers.InputLayer((len(xs_test), len(xs_test[0])), input_var=input_var)
    l_recur_a = las.layers.RecurrentLayer(l_in, num_units= 50)
    l_hidden = las.layers.DenseLayer(l_recur_a, num_units = 4,nonlinearity = las.nonlinearities.softmax, W=las.init.Normal(0.1))
    l_recur_b = las.layers.RecurrentLayer(l_hidden, num_units = 4) #Try doing custom
    # -----pure classes below
    c_l_in = las.layers.InputLayer
    c_l_recur_a = las.layers.RecurrentLayer
    c_l_hidden = las.layers.DenseLayer
    c_l_recur_b = las.layers.RecurrentLayer #Try doing custom
    c_output = las.layers.DenseLayer
    #layers = [('input', c_l_in), ('a', c_l_recur_a), ('h', c_l_hidden), ('b', c_l_recur_b),('output', c_output)]
    layers = [('input', c_l_in), ('h', c_l_hidden), ('h2', c_l_hidden), ('output', c_output)]


    #o = binary_hinge_loss
    net0 = NeuralNet(layers=layers,
                     regression = True,
                     y_tensor_type=  theano.tensor.type.TensorType('float64', (False,True)) ,
                 input_shape = (None, num_features),
#                  input_input_var = input_var,
#                  a_num_units = 50,
                  h_num_units = 200,
                  h_nonlinearity =  las.nonlinearities.softmax, 
                  h_W = las.init.Normal(0.1), #experiment
                  h2_num_units = 200,
                  h2_nonlinearity =  las.nonlinearities.softmax, 
#                  b_num_units = 4,
                 output_num_units=50,
                 output_nonlinearity=softmax,
                 
                 objective_loss_function = loss_function, #vs squared_loss or custom function
                 update=nesterov_momentum,
                 update_learning_rate=0.3,
                 update_momentum=0.1,
                 
                 
                 train_split=nolearn.lasagne.TrainSplit(eval_size=0.2),
                 verbose=1,
                 max_epochs=5)
    print "Begin training"
    net0.fit(xs_train, ys_train)
    
    
    
    
#     l_out = l_recur_b
#     
#     predictions = las.layers.get_output(l_out)
#     loss = las.objectives.squared_error(predictions, target_var)
#     #loss = las.objectives.aggregate(loss)
#     loss=loss.mean()
#     
#     params = las.layers.get_all_params(l_out, trainable =True)
#     
#     updates_sgd = las.updates.sgd(loss, params, learning_rate=0.0001)
#     updates = las.updates.apply_momentum(updates_sgd, params, momentum=0.9)
#     
#     
#     
#     test_prediction = las.layers.get_output(l_out, deterministic=True)
#     test_loss = las.objectives.squared_error(test_prediction, target_var)
#     test_loss = test_loss.mean()
#     # As a bonus, also create an expression for the classification accuracy:
#     test_acc = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_prediction, axis=1), target_var),
#                       dtype=theano.config.floatX) # SHOULD BE REVIEWED TODO 
# 
#     # Compile a function performing a training step on a mini-batch (by giving
#     # the updates dictionary) and returning the corresponding training loss:
#     train_fn = theano.function([input_var, target_var], loss, updates=updates)    
#     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
#     
#     print("Starting training\n---------")
#     
#     for epoch in range(500):
#         # In each epoch, we do a full pass over the training data:
#         train_err = 0
#         train_batches = 0
#         start_time = time.time()
#         for batch in iterate_minibatches(xs_train, ys_train, 500, shuffle=True):
#             print batch[1]
#             inputs, targets = batch
#             train_err += train_fn(inputs, targets)
#             train_batches += 1
# 
#         # And a full pass over the validation data:
#         val_err = 0
#         val_acc = 0
#         val_batches = 0
#         for batch in iterate_minibatches(xs_test, ys_test, 500, shuffle=False):
#             inputs, targets = batch
#             err, acc = val_fn(inputs, targets)
#             val_err += err
#             val_acc += acc
#             val_batches += 1
# 
#         # Then we print the results for this epoch:
#         print("Epoch {} of {} took {:.3f}s".format(
#             epoch + 1, num_epochs, time.time() - start_time))
#         print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#         print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#         print("  validation accuracy:\t\t{:.2f} %".format(
#             val_acc / val_batches * 100))
# 
#     # After training, we compute and print the test error:
#     test_err = 0
#     test_acc = 0
#     test_batches = 0
#     for batch in iterate_minibatches(xs_test, ys_test, 500, shuffle=False):
#         inputs, targets = batch
#         err, acc = val_fn(inputs, targets)
#         test_err += err
#         test_acc += acc
#         test_batches += 1
#     print("Final results:")
#     print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#     print("  test accuracy:\t\t{:.2f} %".format(
#         test_acc / test_batches * 100))
#     

x, y , scaler = read_file_building()
encoder = LabelEncoder()
y = encoder.fit_transform(y).astype(np.int32)
y_scaler = StandardScaler()# MinMaxScaler((0,10.0))
y = y_scaler.fit_transform(y)
#y += #NEED TO MAKE IT SO NO NEGATIVE

build_net(*partition_data(x, y), y_scaler = y_scaler)
    

    