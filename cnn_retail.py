# import
import tensorflow as tf
import input_data as data
import pb2scipy as pb
import random
import time
import util_mlp as ut
import glob
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
start_time = time.time()

#get arguments
args=ut.GetInputArguments().parse_args()
train_file_original=glob.glob(args.references_in)[0] #training file
test_file=glob.glob(args.queries_in)[0] # testing file
attribute_index = glob.glob(args.attribute_index)[0]
hidden_layers0 = [875]
#training data
train_file = "./shuf_train_data_cnn.pb" #new file created containing the shuffled data
#cmnd = "head -n 1 "+train_file_original+"> "+train_file+" && tail -n +2 "+train_file_original+" | shuf >> "+train_file
#os.system(cmnd) #shuffle the training data file
gen_valid = pb.Pb2Scipy("./data/clean_test_10000.pb",[0,1,2],2500)
target_valid=np.array([])
for mat in gen_valid:
    X_valid=mat[0].todense().astype(np.float32)
    target_valid=np.hstack((target_valid,np.array(mat[2])))
n_input =  data.file_len(attribute_index)+1 # kiabi data input (number of features)
n_classes = 1 # it's a regression problem
# Parameters
num_steps = 2000
learning_rate = 1e-3
training_iters = 100000
batch_size = 80
display_step = 10

# Network Parameters
#n_input = X_train.shape[1] # KIABI data input (number of feature)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
def compute_rmse(_actuals,_preds):
    _maximum = tf.maximum(1.,tf.reduce_sum(tf.square(_actuals)))
    return 100*tf.sqrt(tf.reduce_sum( tf.square(_actuals -_preds) )/ _maximum)
def compute_wape(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.abs(tf.sub(_preds,_actuals)))/divi_reduce
def compute_bias_square(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.square(tf.sub(_preds,_actuals)))/divi_reduce
def compute_bias(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.sub(_preds,_actuals))/divi_reduce

# Create model
def conv2d(obs, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(obs, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(obs, k):
    return tf.nn.max_pool(obs, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout,_n_input):
    
    # Reshape input observation 42*42=1722
    #_X = tf.reshape(_X, shape=[-1, 42, 42, 1])
    _X = tf.reshape(_X, shape=[-1, _n_input, 1, 1])
    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=1) # 42*42*nb_output1
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=3) # 14*14*nb_output2
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2,1.)# _dropout)


    #Convolution Layer*
    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    #Max pooling (down_sampling)
    conv3 = max_pool(conv3,k=2) #7*7*nb_output3
    #Apply Dropout
    conv3 = tf.nn.dropout(conv3, _dropout)

    #Convolution Layer*
    conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'])
    #Max pooling (down_sampling)
    #conv4 = max_pool(conv4,k=1) #7*7*nb_output4
    #Apply Dropout
    conv4 = tf.nn.dropout(conv4, _dropout)

    # Fully connected layer 1
    dense1 = tf.reshape(conv4, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv5 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias
weights0={}
biases0 = {}
alpha0 = {}
weights0["h1"] = tf.Variable(tf.random_normal([n_input, hidden_layers0[0]],stddev=np.sqrt(2.0/hidden_layers0[0])))
biases0["b1"] = tf.Variable(tf.zeros([hidden_layers0[0]]))
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 1, 1, 3],stddev=np.sqrt(2.0/3))), # 5x5 conv, 1 input, 3 outputs
    'wc2': tf.Variable(tf.random_normal([5, 1, 3, 5],stddev=np.sqrt(2.0/3))), # 5x5 conv, 3 inputs, 5 outputs
    'wc3': tf.Variable(tf.random_normal([5, 1, 5, 7],stddev=np.sqrt(2.0/3))), # 5x5 conv, 5 inputs , 7 outputs *
    'wc4': tf.Variable(tf.random_normal([2, 1, 7, batch_size],stddev=np.sqrt(2.0/batch_size))), #2x2 conv , 7 inputs , 10 outputs *
    'wd1': tf.Variable(tf.random_normal([11680*batch_size, 458],stddev=np.sqrt(2.0/458))), # fully connected, 7*7*1050 inputs, 5 outputs
    #'wd2': tf.Variable(tf.random_normal([7*7*10, 5])), # fully connected, 11*11*1050 inputs, 5 outputs
    #'wd3': tf.Variable(tf.random_normal([7*7*1050, 5])), # fully connected, 11*11*1050 inputs, 5 outputs
    'out': tf.Variable(tf.random_normal([458, 1])) # 20 inputs, 1 outputs (forecast)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([3])),
    'bc2': tf.Variable(tf.random_normal([5])),
    'bc3': tf.Variable(tf.random_normal([7])),
    'bc4': tf.Variable(tf.random_normal([batch_size])),
    #'bc5': tf.Variable(tf.random_normal([5])),
    'bd1': tf.Variable(tf.random_normal([458])),
    #'bd2': tf.Variable(tf.random_normal([5])),
    #'bd3': tf.Variable(tf.random_normal([5])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Construct model
trans0 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, weights0['h1'],a_is_sparse=True)+ biases0['b1']),keep_prob)
pred = conv_net(trans0, weights, biases, keep_prob,hidden_layers0[-1])

# Define loss and optimizer
#rmse = tf.reduce_mean(tf.square(y-pred) )
wape =compute_wape(y,pred)
bias = compute_bias(y,pred)
rmse = compute_rmse(y,pred)
step = tf.Variable(0, trainable=False)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rmse,global_step=step)


# Initializing the variables
init = tf.initialize_all_variables()
#Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # lists' initialized to plot later
    Bias_list = []
    wape_list = []
    Rmse_list = []
    valid_bias = [1000]
    valid_wape = []
    valid_rmse = []
    #training with RMSE as objective function
    for i in xrange(1):
        print "rebelote"
        early_stopping = False
        gen_train = pb.Pb2Scipy(train_file,[0,1,2],batch_size)
       
        for mat in gen_train:
        
            skip_batch = random.randrange(0,2) #randomly decide to skip or keep the current batch
            #if not skip_batch:
             #   continue
        
            batch_xs=mat[0].todense().astype(np.float32)
            batch_ys=np.array(mat[2])
        
            _, rmseRes,biasRes,wapeRes,predRes,step_now = sess.run([optimizer, rmse, bias , wape,pred,step] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes,biasRes,wapeRes,step_now,crentroRes = sess.run([optimizer4, rmseF, bias , wape,step,cross_entropy] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            print "rmse_train [%s] =  %s | bias_train = %s  | wape = %s  " % (step_now,rmseRes,biasRes,wapeRes)

            if step_now % 200==0 :
                #print "rmse_train [%s] =  %s | bias_train = %s  | wape = %s  | rate = %s" % (step_now,rmseRes,biasRes,wapeRes,rateRes)
        
                #pred_valid = np.empty((0,1))
                pred_valid = np.array([])
                gen_valid = pb.Pb2Scipy("./data/clean_test_10000.pb",[0,1,2],batch_size)
                
                for mat in gen_valid:
                    X_valid=mat[0].todense().astype(np.float32)
                    pred_valid = np.hstack((pred_valid, np.array(tf.reshape(tf.nn.relu(pred),[-1]).eval({x: X_valid, keep_prob:1.}))))
                    
                metrics_valid = ut.GenerateMetrics(target_valid, pred_valid)
                valid_bias.append(float(metrics_valid["bias"]))
                valid_wape.append(float(metrics_valid["wape"]))
                valid_rmse.append(float(metrics_valid["rmse"]))
                print " for validation set:    %s   |  %s    |  %s " % (float(metrics_valid["rmse"]),float(metrics_valid["bias"]),float(metrics_valid["wape"]))
            #if (abs(valid_bias[-1]) <6. and (valid_wape[-1])<106) and i>min_iter :
             #       early_stopping = True
              #      break
        print "end ", i
        print "excution time after iter: ",i, ("--- %s seconds ---" % (time.time() - start_time))
        if early_stopping :
            break

    print "Optimization Finished!"
