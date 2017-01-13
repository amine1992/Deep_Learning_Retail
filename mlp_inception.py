# imports
import input_data as data
import numpy as np
import tensorflow as tf
import pb2scipy as pb
import random
import time
import util_mlp as ut
import glob
import sys
import matplotlib.pyplot as plt
import os
start_time = time.time()

#get arguments
args=ut.GetInputArguments().parse_args()
train_file_original=glob.glob(args.references_in)[0] #training file
test_file=glob.glob(args.queries_in)[0] # testing file
attribute_index = glob.glob(args.attribute_index)[0]


# Parameters
learning_rate= args.learning_rate 
training_epochs = args.training_epochs
batch_size = args.batch_size
dropout = .75 # Dropout, probability to keep units
min_iter = -1

# Network layers' Parameters
model_parameters_in = args.model_parameters_in
hidden_layers0 = [500,12]
hidden_layers1 = map(int,model_parameters_in[1:-1].split(","))
hidden_layers2 = [191,87,3]
hidden_layers3 = [47,13,4]
hidden_layers4 = [101,57,2]
hidden_layers5 = [77,28]
hidden_layers6 = [97,66]
hidden_layers7 = [87,23]
hidden_layers8 = [47,18]
hidden_layers9 = [101,57]
hidden_layers10 = [27,8]


n_input =  data.file_len(attribute_index) # data input (number of features)
n_classes = 1 # it's a regression problem


#input data
#training data
train_file = "./shuf_train_data_mlp_multimodal.pb" #new file created containing the shuffled data
#cmnd = "head -n 1 "+train_file_original+"> "+train_file+" && tail -n +2 "+train_file_original+" | shuf >> "+train_file
#os.system(cmnd) #shuffle the training data file
 #create a generator for the training data

#Cross-validation data
gen_valid = pb.Pb2Scipy("./data/GRMEN/clean_test_10000.pb",[0,1,2],2500)
target_valid=np.array([])
for mat in gen_valid:
    X_valid=mat[0].todense().astype(np.float32)
    target_valid=np.hstack((target_valid,np.array(mat[2])))
#test data
gen_test = pb.Pb2Scipy(test_file,[0,1,2],2500)


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# define parameters
def init_weights(_hidden_layers,_n_input):
    _weights = {}
    _weights['h1']=tf.Variable(tf.random_normal([_n_input, _hidden_layers[0]],stddev=np.sqrt(2.0/_hidden_layers[0]))) #first matrice
    for i in xrange(2,len(_hidden_layers)+1):
        _weights['h'+str(i)]= tf.Variable(tf.random_normal([_hidden_layers[i-2], _hidden_layers[i-1]],stddev=np.sqrt(2.0/_hidden_layers[i-1])))
    _weights['out']=tf.Variable(tf.random_normal([_hidden_layers[-1], 1],stddev=np.sqrt(2.0/_hidden_layers[-1]))) #matrice between last layer and output
    return _weights
def init_biases(_hidden_layers):
    _biases = {}
    _biases['b1'] = tf.Variable(tf.zeros([_hidden_layers[0]]))
    for i in xrange(2,len(_hidden_layers)+1):
        _biases['b'+str(i)] = tf.Variable(tf.zeros([_hidden_layers[i-1]]))
    _biases['out']= tf.Variable(tf.zeros([1]))
    return _biases

# for PRELU activation funciton, we have to define the alpha for every layer
def init_alpha (_hidden_layers):
    _alpha = {}
    _alpha ['b1'] = tf.Variable(tf.zeros([1]))
    for i in xrange(2,len(_hidden_layers)+1):
        _alpha['b'+str(i)] = tf.Variable(tf.zeros([1]))
    _alpha['out']= tf.Variable(tf.zeros([1]))
    return _alpha

# define the functions to compute the metrics
def compute_rmse(_actuals,_preds):
    _maximum = tf.maximum(1.,tf.reduce_sum(tf.square(_actuals)))
    return 100*tf.sqrt(tf.reduce_sum( tf.square(_actuals -_preds) )/ _maximum)
def compute_wape(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.abs(tf.sub(_preds,_actuals)))/divi_reduce
def compute_bias_square(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.square(tf.sub(_preds,_actuals))/divi_reduce)
def compute_bias(_actuals,_preds):
    divi_reduce = tf.maximum(1.,tf.reduce_sum(_actuals))
    return 100*tf.reduce_sum(tf.sub(_preds,_actuals))/divi_reduce

# Activation function
def prelu (_X, _alpha):
    #PReLu: parametric Relu
    return tf.maximum(0.0, _X) + tf.mul(_alpha, tf.minimum(0.0, _X))
#L2Reg 
def l2reg (_weights):
    l2regRes = 0
    for weight in _weights.values():
        l2regRes = tf.reduce_sum(l2regRes+tf.nn.l2_loss(weight))
    return 0.001*l2regRes

# Create model
def multilayer_perceptron(_X, _weights, _biases, _dropout,_alpha):
    _n_layers = len(_weights)
    layer_begin = tf.nn.dropout(_X, 1.)
    #feedforward propagation
    for layer in xrange(1,_n_layers):
        #_weights['h'+str(layer)] = _weights['h'+str(layer)] - _sigma*_weights['h'+str(layer)]
        layer_begin = tf.matmul(layer_begin, _weights['h'+str(layer)],a_is_sparse=True)+ _biases['b'+str(layer)]
        layer_begin = prelu(layer_begin,_alpha['b'+str(layer)])
        layer_begin = tf.nn.dropout(layer_begin,_dropout) 
    
    return tf.reshape(tf.add(tf.matmul(layer_begin, _weights['out']), _biases['out']),[-1])  
 
# Store layers weight & bias & parameter for relu for sub_net1
weights0={}
biases0 = {}
alpha0 = {}
weights0["h1"] = tf.Variable(tf.random_normal([n_input, hidden_layers0[0]],stddev=np.sqrt(2.0/hidden_layers0[0])))
biases0["b1"] = tf.Variable(tf.zeros([hidden_layers0[0]]))
alpha0["b1"] = tf.Variable(tf.zeros([1])) 
# Store layers weight & bias & parameter for relu for sub_net1
weights1 = init_weights(hidden_layers1,n_input)
biases1 = init_biases(hidden_layers1)
alpha1 = init_alpha(hidden_layers1) 

# Store layers weight & bias & parameter for relu for sub_net2
weights2 = init_weights(hidden_layers2,n_input)
biases2 = init_biases(hidden_layers2)
alpha2 = init_alpha(hidden_layers2)

# Store layers weight & bias & parameter for relu for sub_net3
weights3 = init_weights(hidden_layers3,n_input)
biases3 = init_biases(hidden_layers3)
alpha3 = init_alpha(hidden_layers3)

# Store layers weight & bias & parameter for relu for sub_net4
weights4 = init_weights(hidden_layers4,n_input)
biases4 = init_biases(hidden_layers4)
alpha4 = init_alpha(hidden_layers4)

# Store layers weight & bias & parameter for relu for sub_net5
weights5 = init_weights(hidden_layers5,n_input)
biases5 = init_biases(hidden_layers5)
alpha5 = init_alpha(hidden_layers5)

# Store layers weight & bias & parameter for relu for sub_net6
weights6 = init_weights(hidden_layers6,n_input)
biases6 = init_biases(hidden_layers6)
alpha6 = init_alpha(hidden_layers6)
# Store layers weight & bias & parameter for relu for sub_net7
weights7 = init_weights(hidden_layers7,n_input)
biases7 = init_biases(hidden_layers7)
alpha7 = init_alpha(hidden_layers7)

# Store layers weight & bias & parameter for relu for sub_net8
weights8 = init_weights(hidden_layers8,n_input)
biases8 = init_biases(hidden_layers8)
alpha8 = init_alpha(hidden_layers8)

# Store layers weight & bias & parameter for relu for sub_net9
weights9 = init_weights(hidden_layers9,n_input)
biases9 = init_biases(hidden_layers9)
alpha9 = init_alpha(hidden_layers9)

# Store layers weight & bias & parameter for relu for sub_net10
weights10 = init_weights(hidden_layers10,n_input)
biases10 = init_biases(hidden_layers10)
alpha10 = init_alpha(hidden_layers10)

w1 = tf.Variable(tf.ones([1]))
w2 = tf.Variable(tf.ones([1]))
w3 = tf.Variable(tf.ones([1]))
w4 = tf.Variable(tf.ones([1]))
w5 = tf.Variable(tf.ones([1]))
w6 = tf.Variable(tf.ones([1]))
w7 = tf.Variable(tf.ones([1]))
w8 = tf.Variable(tf.ones([1]))
w9 = tf.Variable(tf.ones([1]))
w10 = tf.Variable(tf.ones([1]))

w_12345 = tf.Variable(tf.ones([1]))
w_678910 = tf.Variable(tf.ones([1]))
wReg = tf.Variable(tf.ones([1]))
biasF = tf.Variable(tf.zeros([1]))
bias_12345 = tf.Variable(tf.zeros([1]))
bias_678910 = tf.Variable(tf.zeros([1]))
biasReg = tf.Variable(tf.zeros([1]))


   
# generate forecast
pred1 = multilayer_perceptron(x, weights1, biases1, keep_prob,alpha1)
pred2 = multilayer_perceptron(x, weights2, biases2, keep_prob,alpha2)
pred3 = multilayer_perceptron(x, weights3, biases3, keep_prob,alpha3)
pred4 = multilayer_perceptron(x, weights4, biases4, keep_prob,alpha4)
pred5 = multilayer_perceptron(x, weights5, biases5, keep_prob,alpha5)
pred6 = multilayer_perceptron(x, weights6, biases6, keep_prob,alpha6)
pred7 = multilayer_perceptron(x, weights7, biases7, keep_prob,alpha7)
pred8 = multilayer_perceptron(x, weights8, biases8, keep_prob,alpha8)
pred9 = multilayer_perceptron(x, weights9, biases9, keep_prob,alpha9)
pred10 = multilayer_perceptron(x, weights10, biases10, keep_prob,alpha10)
pred_12345 = tf.mul(pred1,w1)+tf.mul(pred2,w2)+tf.mul(pred3,w3)+tf.mul(pred4,w4)+tf.mul(pred5,w5) + bias_12345
pred_678910 = tf.mul(pred6,w6)+tf.mul(pred7,w7)+tf.mul(pred8,w8)+tf.mul(pred9,w9)+tf.mul(pred10,w10)+ bias_678910
predF = tf.mul(pred_12345,w_12345)+tf.mul(pred_678910,w_678910) + biasF

#Instantiate loss functions
divi_reduce = tf.maximum(1.,tf.reduce_sum(y))
wape =compute_wape(y,predF)
bias = compute_bias(y,predF)
bias_train = compute_bias_square(y,predF)
rmse1 = compute_rmse(y,pred1) 
rmse2 = compute_rmse(y,pred2)
rmse3 = compute_rmse(y,pred3) 
rmse4 = compute_rmse(y,pred4)
rmse5 = compute_rmse(y,pred5)
rmse6 = compute_rmse(y,pred6)
rmse7 = compute_rmse(y,pred7) 
rmse8 = compute_rmse(y,pred8)
rmse9 = compute_rmse(y,pred9)
rmse10 = compute_rmse(y,pred10)
rmse_12345 =  compute_rmse(y,pred_12345)
rmse_678910 = compute_rmse(y,pred_678910)
rmseF = compute_rmse(y,predF)
biasF = compute_bias_square(y,predF)


# Define optimizers
step = tf.Variable(0, trainable=False)
rate1 = tf.train.exponential_decay(0.008, step, 5, .9999)
rate2 = tf.train.exponential_decay(1e-4, step, 5, .9998)
optimizer1 = tf.train.AdamOptimizer(5e-3).minimize(rmse1)
optimizer2 = tf.train.AdamOptimizer(5e-3).minimize(rmse2)
optimizer3 = tf.train.AdamOptimizer(5e-3).minimize(rmse3)
optimizer4 = tf.train.AdamOptimizer(1e-4).minimize(rmse4)
optimizer5 = tf.train.AdamOptimizer(1e-4).minimize(rmse5)
optimizer6 = tf.train.AdamOptimizer(1e-4).minimize(rmse6)
optimizer7 = tf.train.AdamOptimizer(5e-3).minimize(rmse7)
optimizer8 = tf.train.AdamOptimizer(1e-4).minimize(rmse8)
optimizer9 = tf.train.AdamOptimizer(1e-4).minimize(rmse9)
optimizer10 = tf.train.AdamOptimizer(1e-4).minimize(rmse10)
optimizer_12345 = tf.train.AdamOptimizer(1e-4).minimize(rmse_12345)
optimizer_678910 = tf.train.AdamOptimizer(1e-4).minimize(rmse_678910)
#optimizer = tf.train.AdamOptimizer(5e-3).minimize(rmse) #,var_list=[w1,w2,w3,w4,w5,w6,biasF]
optimizerF = tf.train.AdamOptimizer(5e-3).minimize(biasF,global_step=step)
#optimizer1 = tf.train.GradientDescentOptimizer(rate).minimize(rmse,global_step=step)

# Initializing the variables
init = tf.initialize_all_variables()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Launch the graph
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
    for i in xrange(2):
        print "new epoch"
        early_stopping = False
        gen_train = pb.Pb2Scipy(train_file,[0,1,2],70)
       
        for mat in gen_train:
        
            skip_batch = random.randrange(0,2) #randomly decide to skip or keep the current batch
            #if not skip_batch:
             #   continue
        
            batch_xs=mat[0].todense().astype(np.float32)
            batch_ys=np.array(mat[2])
           
        #training
            #_, rmseRes1,biasRes1,wapeRes1, predRes1 = sess.run([optimizer1, rmse1, bias , wape,pred1] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes2,biasRes2,wapeRes2, predRes2 = sess.run([optimizer2, rmse2, bias , wape,pred2] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes3,biasRes3,wapeRes3, predRes3 = sess.run([optimizer3, rmse3, bias , wape,pred3] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes4,biasRes4,wapeRes4, predRes4 = sess.run([optimizer4, rmse4, bias , wape,pred4] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes5,biasRes5,wapeRes5, predRes5 = sess.run([optimizer5, rmse5, bias , wape,pred5] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes6,biasRes6,wapeRes6, predRes6 = sess.run([optimizer6, rmse6, bias , wape,pred6] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes7,biasRes7,wapeRes7, predRes7 = sess.run([optimizer7, rmse7, bias , wape,pred7] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes8,biasRes8,wapeRes8, predRes8 = sess.run([optimizer8, rmse8, bias , wape,pred8] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes9,biasRes9,wapeRes9, predRes9 = sess.run([optimizer9, rmse9, bias , wape,pred9] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            #_, rmseRes10,biasRes10,wapeRes10, predRes10 = sess.run([optimizer10, rmse10, bias , wape,pred10] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )

            _, rmseRes_12345,biasRes,wapeRes,predRes = sess.run([optimizer_12345, rmse_12345, bias , wape,pred_12345] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            _, rmseRes_678910,biasRes,wapeRes,predRes = sess.run([optimizer_678910, rmse_678910, bias , wape,pred_678910] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            _, rmseRes,biasRes,wapeRes,step_now = sess.run([optimizerF, biasF, bias , wape,step] , feed_dict={x: batch_xs, y: batch_ys, keep_prob : dropout} )
            
            #Validation
            if step_now % 200==0  :
                #print "rmse_train [%s] =  %s | bias_train = %s  | wape = %s  | rate = %s" % (step_now,rmseRes,biasRes,wapeRes,rateRes)
        
                #pred_valid = np.empty((0,1))
                pred_valid = np.array([])
                gen_valid = pb.Pb2Scipy("./data/GRMEN/clean_test_10000.pb",[0,1,2],2500)
                
                for mat in gen_valid:
                    X_valid=mat[0].todense().astype(np.float32)
                    pred_valid = np.hstack((pred_valid, np.array(tf.reshape(tf.nn.relu(predF),[-1]).eval({x: X_valid, keep_prob:1.}))))
                    
                metrics_valid = ut.GenerateMetrics(target_valid, pred_valid)
                valid_bias.append(float(metrics_valid["bias"]))
                valid_wape.append(float(metrics_valid["wape"]))
                valid_rmse.append(float(metrics_valid["rmse"]))
                print " for validation set:    %s   |  %s    |  %s " % (float(metrics_valid["rmse"]),float(metrics_valid["bias"]),float(metrics_valid["wape"]))
            #if (abs(valid_bias[-1]) <4. and (valid_wape[-1])<108)  :
             #       early_stopping = True
              #      break
        print "end ", i
    
        if early_stopping :
            break
    print "excution time after training", ("--- %s seconds ---" % (time.time() - start_time))
        #print "rmse_train [%s] =  %s | bias_train = %s  | wape = %s " % (step.eval(),rmseRes,biasRes,wapeRes)
        #if step_rmse > training_epochs:
         #   break
    gen_train1 = pb.Pb2Scipy(train_file_original,[0,1,2],2000)
    
    fitted = np.array([])
    actuals = np.array([])
    with open('./results/res2/fitted_data_LR', 'w') as f:
        for mat in gen_train1:
            dummy_fitted = np.array([])
            batch_xs=mat[0].todense().astype(np.float32)
            actuals=np.hstack((actuals,np.array(mat[2])))
            dummy_fitted = np.array(tf.reshape(tf.nn.relu(predF),[-1]).eval({x: batch_xs, keep_prob:1.})) 
            fitted = np.hstack((fitted,dummy_fitted))  
            for item in dummy_fitted:
                f.write("%s\n" % item)    
    #testing
    preds = np.array([])
    target = np.array([])
    
    #testing
    gen_test = pb.Pb2Scipy(test_file,[0,1,2],3000)
    with open('./results/res2/forecast_data_LR', 'w') as f:
        for mat in gen_test:
            dummy_preds = np.array([])
            #reformatting the data
            X_test=mat[0].todense().astype(np.float32)
            #regrouping data
            target=np.hstack((target,np.array(mat[2])))
            #getting results
            dummy_preds = np.array(tf.reshape(tf.nn.relu(predF),[-1]).eval({x: X_test, keep_prob:1.}))
            #stacking results
            preds = np.hstack((preds, dummy_preds ))
            for item in dummy_preds:
                    f.write("%s\n" % item)
        #if step_test >500 :
         #   break
    metrics_test = ut.GenerateMetrics(target, preds)
    metrics_train = ut.GenerateMetrics(actuals,fitted)
    print "%s   | %s   |  %s   |  %s  || %s   |  %s    |  %s " % (hidden_layers1,float(metrics_train["rmse"]),float(metrics_train["bias"]),float(metrics_train["wape"]),float(metrics_test["rmse"]),float(metrics_test["bias"]),float(metrics_test["wape"]))
    print "-----------------------------------------------------------------------------------------------------------"
  
   
    save_path = saver.save(sess, "./results/res2/model.ckpt")
print "excution time at the end", ("--- %s seconds ---" % (time.time() - start_time))
