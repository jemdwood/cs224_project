import numpy as np
import lasagne as las
import theano
import time
import json
import re
import pandas as pd
from collections import *
import nolearn
import random
import math
import csv
import collections
import sklearn
from encodings.punycode import digits
from lasagne.nonlinearities import softmax
from lasagne.updates import apply_momentum
from lasagne.updates import nesterov_momentum
#from lasagne.objectives import binary_hinge_loss
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model as linear
from nolearn.lasagne import NeuralNet
from numpy import float64, inf
from scipy.sparse.csc import csc_matrix
from scipy import sparse
from sklearn.preprocessing.label import LabelEncoder
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from scipy.sparse.construct import hstack


class ParamTracker():
    def __init__(self):
        self.write = False
        self.use_indicators = False
        self.years = [ 2005, ]#2006, 2007, 2008, 2009 ]
        self.base_filename ="CtyxCty_US_"
        self.node_base_filename = "node_data_"
        self.test_split = 0.15
        self.y_scaler = MinMaxScaler((-1,1))
        self.x_scaler = None #StandardScaler()
        self.regression_model = linear.LinearRegression()
        self.edge_threshold = 0
        self.indicator_encoder = LabelEncoder()
        self.indicator_minibatches = 100
        self.node_info = self.read_node_info()
        self.county_data = read_county_data('counties.txt')
        
    
    def filter_node_info(self, x):
        to_delete = [
                     'Out_D',
                     'In_D',
                     'D_Centr'
                     ]
        for delete in to_delete:
            if delete in x:
                del x[delete]
        return x
        
    
    def read_node_info(self):
        node_info = dict()
        for year in self.years:
            filename = self.node_base_filename + str(year) + ".txt"
            with open(filename, 'r') as nodef:
                 ninf= json.load(nodef, object_hook = self.filter_node_info)   
                 node_info[year] = ninf 
        return node_info

#class to incorporate all the data fields for counties
class County:
    state = 0
    fips = 0
    name = 0
    countySeat = 0
    population = 0
    landAreaKM = 0
    landAreaMI = 0
    waterAreaKM = 0
    waterAreaMI = 0
    totalAreaKM = 0
    totalAreaMI = 0
    latitude = 0
    longitude = 0

def read_county_data(f):
    county_dict = {}
    with open(f) as inputFile:
        lines = inputFile.readlines()
        for line in lines:
            c = County()
            l = line.split()
            c.state = l[1]
            c.fips = str(l[2])
            while len(c.fips) < 6:
                c.fips = '0'+c.fips
            c.name = l[3]
            c.countySeat = l[4]
            c.population = int(l[5])
            c.landAreaKM = float(l[6])
            c.landAreaMI = float(l[7])
            c.waterAreaKM = float(l[8])
            c.waterAreaMI = float(l[9])
            c.totalAreaKM = float(l[10])
            c.totalAreaMI = float(l[11])
            c.latitude = float(l[12])
            c.longitude = float(l[13])
            county_dict[c.fips] = c
    return county_dict

#calculates the linear squared distance between lattidue and longitude points 
#its not the ideal distance, buts its big enough    
def calculate_distance(p1, p2):
    return math.pow(p1.latitude - p2.latitude, 2) + math.pow(p1.longitude - p2.longitude, 2)

def current_population():
    return "1CurPop"

def last_population():
    return "2LastPop"

def get_distance_fn(params):
    def get_distance(info):
        d = read_cur_res(info)
        o = read_last_res(info)
        if d in params.county_data and o in params.county_data:
            dist = calculate_distance(params.county_data[o], params.county_data[d])
        else:
            dist =  1.0*math.sqrt(abs(int(d)-int(o)))
        return dist

    return get_distance
    

def feature_extractions(params = None):
    return {
     current_population(): lambda x: x["cur_pop"], # MUST BE THE FIRST FEATURE
     last_population(): lambda x: x["last_pop"],
     #"3 extra": lambda x: x["extra"],
     "4inv_c_p": lambda x: 1.0/x["cur_pop"],
     #"5 inverse square current pop": lambda x: 1.0/(x["cur_pop"]**2),
     "6inv_l_p": lambda x: 1.0/x["last_pop"],
     #"7 inverse square last pop": lambda x: 1.0/(x["last_pop"]**2),
     "8prod": lambda x: x["last_pop"] * x["cur_pop"],
     "9norm" : lambda x: math.sqrt(x["last_pop"]**2 + x["cur_pop"]**2),
     "dist" : lambda x: get_distance_fn(params)(x),
     "dist2" : lambda x: get_distance_fn(params)(x)**2
    }

def scale_range():
    return (0.0, 5.0)

def preprocessed_data(year):
    return "preprocessed_data_%i_json.txt" % (year) #_less_features

def featurize(i, params):
    featurize_fns = map(lambda x: (x[0], x[1]), sorted(feature_extractions(params).items(), key = lambda x: x[0]))
    sparse_features = defaultdict(lambda: 0)
    map(lambda fn: sparse_features.update({fn[0]: fn[1](i)}), featurize_fns)
    return sparse_features

# res moved to
def read_cur_res(info):
    return str(info["curr_res"])

#place moved from
def read_last_res(info):
    return str(info["last_res"])

def read_movers(info):
    return int(info['movers'])

def read_uniq_key(info_dict):
    return int(read_cur_res(info_dict) + read_last_res(info_dict))

def supplement(info):
    #add info about city 
    pass
    # TODO
    return info

def write_preprocessed_json(data, year):
    print "Writing JSON for %i" % ( year )
    with open(preprocessed_data(year), "wb") as jsonfile:
        jsonfile.write('[')
        for r, datum in enumerate(data):
            json.dump(datum, jsonfile)
            if (1.0*(r+1.0)/(len(data))) %0.01 < 0.0000005:
                print "%0.2f JSON written" % (1.0*(r+1.0)/(len(data))) 
            if r < len(data) - 1:
                jsonfile.write(',')
            else:
                print 'finished'
            
        jsonfile.write(']')
        jsonfile.close()

def read_info(f):
    infos = []
    a = 0 #error counting
    county_travel_vector = dict() # TODO NEED TO FIT AND SCALE BEFORE THIS IS ADDED TO TEH X VECTOR, THESE VALUES SHOULD NOT BE SCALED!!!!!!!
    for line in f:
        try:
            d = extract_line_info(line)
        except:
            a += 1
            continue
        infos.append(d)
        county_travel_vector[read_cur_res(d)] = True#Adds feature for destination and origin
        county_travel_vector[read_last_res(d)] = True #Adds feature for destination and origin
    print "%i samples not processed out of %i (%f)" % (a, len(infos), 1.0*a/len(infos))    
    infos.sort(key = read_uniq_key)
    return supplement(infos), county_travel_vector.keys()

def pack_county(c):
    return "C_" + str(c)

def unpack_county(pc):
    return pc[2:]

def append_label_to_keys(dic, label):
    new_d = dict()
    for k in dic:
        new_d[label+k] = dic[k]
    return new_d

def node_graph_info(year, i, params):
    node_info = dict()
    dest_node_info = params.node_info[year][read_cur_res(i)]
    node_info.update(append_label_to_keys(dest_node_info, 'D_'))
    
    origin_node_info = params.node_info[year][read_last_res(i)]
    node_info.update(append_label_to_keys(origin_node_info, 'O_'))
    return node_info
    
    

def write_data(paramsT):
    data = list()
    for year in paramsT.years:
        print "processing year: %i" %(year)
        f = open(paramsT.base_filename+str(year)+".txt", "r+") #TODO need to add in the years!!!!!!!!!!!!!!
        infos, all_counties = read_info(f)
        random.shuffle(infos) #randomize order in year so that we see different configurations
        createGraphAndCheck(infos, year, params)
        for n, i in enumerate(infos):
            base_features = featurize(i, paramsT)
            base_features['dest'] = pack_county(read_cur_res(i))
            base_features['origin'] = pack_county(read_last_res(i))
            base_features['year'] = year
            base_features['output'] = read_movers(i)
            base_features.update(node_graph_info(year, i, paramsT))            
            data.append(base_features)
            if (1.0*n/len(infos))%0.03 < 0.000001:
                print "%0.2f info processed" % (1.0*(n+1.0)/len(infos)) 
        f.close()
        write_preprocessed_json(data, year)
    print "Processed data written"

def read_all_files(params):
    #X = csc_matrix
    xs = []
    ys = []
    years = []
    for year in params.years:
        try:
            x_, y_, years_ = read_file_building(year)
            xs.append(x_)
            ys.append(y_)
            years.append(years_)
        except MemoryError:
            "Could not finish %i" % (year)
    return pd.concat(xs), pd.concat(ys), np.concatenate(years)
        

def read_file_building(year):
    print "Reading JSON for %i" % ( year )
    df = pd.read_json(preprocessed_data(year))
    print "Input complete"
    y = df.pop('output').astype(np.float64)
    years = df.pop('year').astype(np.int16).as_matrix()
    X = df.fillna(value=0)
    print "Data read for %i" % ( year )
    return X, y, years

def prep_y(y, X, params):
    x_cur_pops = X[current_population()]
    y = y.div(x_cur_pops) # DIVIDES BY CURRENT POPULATION TO REMOVE THIS FORM COMPUTATION, MIGHT NOT WANT
    y = y.as_matrix()
    y_scaler = params.y_scaler
    y = y_scaler.fit_transform(y)
    return y, y_scaler
    
def pd_arr_iter(pd_row_iter):
    tuple = pd_row_iter.next()
    yield {'origin': tuple[1].values[0], 'dest':tuple[1].values[1]}
    
def merge_dicts(dicts):
    x1, x2 = dicts
    x1.update(x2)
    #print x1
    return x1  

def get_reduce_fn(keys):
    def reduce_fn(row):
        return ({keys[0] : row[0], keys[1]: row[1]})
    return reduce_fn

def transform_into_indicators(origin_dest_arr, params):
    print "Fitting the label encoder"
    keys=  origin_dest_arr.keys()
    origins = map(lambda x: {'origin': x}, origin_dest_arr['origin'].as_matrix().astype(str))
    dests = map(lambda x: {'dest': x}, origin_dest_arr['dest'].as_matrix().astype(str))
    d = DictVectorizer(sort=False)
    reduce_fn = get_reduce_fn(keys)
    print "mapping"
    #cat = origin_dest_arr.to_dict(orient = 'records')
    cat = map(lambda x: x.__self__, origin_dest_arr.apply(reduce_fn, axis = 1, raw = True, reduce = True))
    print "Transforming"
    return d.fit_transform(cat)
    
#     params.indicator_encoder.fit(np.concatenate([origins, dests]))
#     origins = params.indicator_encoder.transform(origins)
#     dests = params.indicator_encoder.transform(dests)
#     print "Transforming the indicators into sparse matrix form"
#     def space_out(origins, dests):
#         batches =params.indicator_minibatches
#         batch_size = len(origins)/batches
#         batch_shape = (batch_size, len(params.indicator_encoder.classes_))
#         sparse_batches = []
#         non_sparse_array = np.ndarray(batch_shape, dtype = np.int) #erased after every batch
#         for b in range(params.indicator_minibatches): 
#             print "\b--running batch %i of %i" %(b+1, batches)
#             for i in range(batch_size):
#                 encoded_origin_x = origins[i]
#                 encoded_dest_x = dests[i]
#                 non_sparse_array[i][encoded_origin_x] = 1
#                 non_sparse_array[i][encoded_dest_x] = -1
#             sparse_batches.append(csc_matrix(non_sparse_array))
#             non_sparse_array = np.ndarray(batch_shape, dtype = np.int)
#         return sparse.vstack(sparse.vstack(sparse_batches))
#             
#     sparse_matrix = space_out(origins, dests)
#     print sparse_matrix


def prep_x(X, params):
    scaler = params.x_scaler
    origin_dest_arr = X.select(lambda x: x == 'origin' or x == 'dest', axis = 1)
    to_scale = X.select(lambda x: x != 'origin' and x != 'dest', axis = 1).astype(np.float64).values
    if scaler != None:
        to_scale = scaler.fit_transform(to_scale)
    indicators = transform_into_indicators(origin_dest_arr, params)
    if params.use_indicators:
        hh = sparse.hstack([to_scale, indicators])
    else:
        hh = sparse.crc(to_scale)
    return hh
 
def extract_line_info(line):
    #HARDCODING TO BE REMOVED
    last_res_pop = int(line[259:267])
    cur_res_pop = int(line[79:87])
    curr_res_code = line[0:6]
    last_res_code = line[6:12] #should be three digits
    movers = int(line[373:380])
    extra = int(line[309:316])
    return {"extra": extra, 'last_pop': last_res_pop, 'cur_pop': cur_res_pop, "curr_res": curr_res_code, "last_res": last_res_code, "movers": movers}

# THIS Needs to be improved to train off of a year at a time and then predict for the next year. TODO cutoff should be based on year not percentage split
def partition_data(xs, ys, params):
    xs = xs.tocsc()
    test_perc = params.test_split
    partition = int((1.0-test_perc)*(xs.shape[0]))
    train = [xs[0:partition], ys[0:partition]]
    test = [xs[partition:], ys[partition:]]
    return train, test


def get_sign(a, b):
    if theano.tensor.eq(a,b):
        return 0.0
    elif theano.tensor.gt(a,b):
        return 1.0
    elif theano.tensor.lt(a,b):
        return -1.0
    else:
        print "ERROR\n\nERROR\n\nRE"

def get_loss_function(scaler):
    def loss_function (a,b):
        if get_sign(a,b) >= 0:
            multiplier = 1.0
        else:
            multiplier = 2.0 
        res = multiplier * np.abs(a-b)# **2#get_sign(transformed_a,transformed_b)*(transformed_a-transformed_b)**2
        return res
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

#shuffles both arrays the same way (index-pairs remain paired just at a different index)
def shuffle(dataset):
    x, y = dataset
    rg = range(len(x))
    random.shuffle(rg)
    return [x[i] for i in rg], [y[i] for i in rg]


# Constructs a map from the list of edges
def constructMap(arr, params):
    edges = defaultdict(list)
    for elem in arr:
        origin = read_last_res(elem)
        dest = read_cur_res(elem)
        edges[origin].append(dest)
    return edges

# Write the map to file in a map-data format
def writeMap(edgeMap, year):
    with open("edge_data_"+str(year)+".txt", 'wb') as outputFile:
        for n1 in edgeMap:
            data = ",".join(edgeMap[n1])
            outputFile.write(n1+","+data+"\n")
        outputFile.close()    
        


# Read map-data from the formatted file
def readMap(year):
    with open("edge_data_"+str(year)+".txt", 'r') as inputFile:
        reader = csv.reader(inputFile, delimiter=",")
        edgeMap = defaultdict(list)
        for line in reader:
            edgeMap[line[0]] = line[1:]
    return edgeMap


# Construct map from array, write to file, 
# then read, reconstruct, and check for accuracy.
def createGraphAndCheck(arr, year, params):
    # Generate map of edges
    edges1 = constructMap(arr,params)

    # Write to file
    writeMap(edges1, year)

    # Read map from file
    edges2 = readMap(year)

    # Check to make sure they're equivalent 
    if edges1 == edges2:
        print "Edge lists match"

    else:
        print "Error: Edge lists don't match."
        print edges1[0]
        print edges2[0]
        assert(False)



def build_net(train, test, y_scaler):

    xs_test, ys_test = test
    xs_train, ys_train = train
    num_features = xs_train.shape[1]
    #assert(num_features == len(feature_extractions().keys()))
    loss_function = get_loss_function(y_scaler)
    
    input_var = theano.tensor.dmatrix('inputs')
    target_var = theano.tensor.dvector('targets')
    
#     l_in = las.layers.InputLayer((len(xs_test), len(xs_test[0])), input_var=input_var)
#     l_recur_a = las.layers.RecurrentLayer(l_in, num_units= 50)
#     l_hidden = las.layers.DenseLayer(l_recur_a, num_units = 4,nonlinearity = las.nonlinearities.softmax, W=las.init.Normal(0.1))
#     l_recur_b = las.layers.RecurrentLayer(l_hidden, num_units = 4) #Try doing custom
    # -----pure classes below
    c_l_in = las.layers.InputLayer
    c_l_recur_a = las.layers.RecurrentLayer
    c_l_hidden = las.layers.DenseLayer
    c_l_recur_b = las.layers.RecurrentLayer #Try doing custom
    c_expression_layer = las.layers.special.ExpressionLayer
    c_output = las.layers.DenseLayer
    #layers = [('input', c_l_in), ('a', c_l_recur_a), ('h', c_l_hidden), ('b', c_l_recur_b),('output', c_output)]
    layers = [('input', c_l_in), ('h', c_l_hidden), ('h2', c_l_hidden),('h3', c_l_hidden),('h4', c_l_hidden), ('output', c_output)]

    print "\nBuilding..."
    #o = binary_hinge_loss
    net0 = NeuralNet(layers=layers,
                     regression=True,
                     y_tensor_type=theano.tensor.type.TensorType('float64', (False, True)) ,
                 input_shape=(None, num_features),
#                  input_input_var = input_var,
#                  a_num_units = 50,
              h_num_units=400,
              
              #h_nonlinearity =  las.nonlinearities.softmax, 
              h2_num_units=50,
              h3_num_units=20,
              h4_num_units=1,
              # h2_nonlinearity =  las.nonlinearities.softmax, 
#                  b_num_units = 4,
                 #e_function=expression_layer_fn,
                 output_num_units=1,
                 # output_nonlinearity=softmax,
                 
                 objective_loss_function=loss_function,  
                 update=nesterov_momentum,
                 update_learning_rate=0.001,
                 update_momentum=0.3,
                 
                 train_split=nolearn.lasagne.TrainSplit(eval_size=0.1),
                 verbose=1,
                 max_epochs=1000)
    print "Begin training"
    net0.fit(xs_train, ys_train)
    print "y: %f" % (ys_test[0])
    print "transformed y: %f" %(y_scaler.inverse_transform([ys_test[0]])[0])
    print "\n"
    print "y: %f" % (ys_test[1])
    print "transformed y: %f" % (y_scaler.inverse_transform([ys_test[1]])[0])

    print "\n predictions: :"
    print "y: {}".format((net0.predict([xs_test[0], xs_test[1]])))
    print y_scaler.inverse_transform(net0.predict([xs_test[0], xs_test[1]]))


#     predicts = net0.predict([[30.0,-1.5,4.5,3087],[1.0,1.0,1.0,1.0],[5.0,0.1,5.0,1000]])
#     print "\nPrediction: %f - 93864 == %f \n %f - 3 == %f \n %f - 1000000 == %f" % (predicts[0], (predicts[0]-93864)*1.0/93864, predicts[1], (predicts[1] - 3)*1.0/3, predicts[2], (predicts[2]-1000000)/1000000)
#     print "\n\nTransformed:"
#     predicts = map(lambda x: y_scaler.inverse_transform([x]), predicts)
#     predicts = [y[0] for y in predicts]
#     print "prediction: %f - 93864 == %f \n %f - 3 == %f \n %f - 1000000 == %f" % (predicts[0], (predicts[0]-93864)*1.0/93864, predicts[1], (predicts[1] - 3)*1.0/3, predicts[2], (predicts[2]-1000000)/1000000)
    print "\n Scores:"
    print "test score: %f" % (net0.score(xs_test,ys_test))
    print net0.score(xs_train,ys_train)
    print "random score: %f" % (net0.score(xs_test,ys_train[0:len(xs_test)]))
    print "random score: %f" % (net0.score(xs_train[0:len(ys_test)],ys_test))
    print net0.layers
    
def regression_model(train, test, weights, y_scaler, params):
      xs_test, ys_test = test
      xs_train, ys_train = train
      clf = params.regression_model
      clf.fit(xs_train, ys_train, weights)
      print "Score: %0.5f \n" % ( clf.score(xs_test, ys_test))

def weight_years(years):
    min_y = min(years)
    max_y = max(years)
    return [((1.0*y-min_y)/max_y) for y in years]
    
    

params = ParamTracker()
if params.write:
    write_data(params)
x, y, years  = read_all_files(params)
print "Prepping data"
y, y_scaler = prep_y(y, x, params)
x = prep_x(x, params)
print "Splitting data"
train, test = partition_data(x,y, params)
weights = weight_years(years[0:len(train)])
regression_model(train, test, weights, y_scaler, params)
#build_net(train, test, y_scaler = y_scaler)
    

    
