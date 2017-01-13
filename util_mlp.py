import argparse
import numpy as np
import matplotlib.pyplot as plt
import pb2scipy as pb

def GetInputArguments():
    parser = argparse.ArgumentParser(description='mlp-tensorflow-forecaster')

    parser.add_argument('--attribute_index', dest='attribute_index', action='store',
                    default="", help="file with the attributes and types")
    parser.add_argument("--references_in", dest="references_in", action="store",
                    default="", help="the paperboat file with the training data")
    parser.add_argument("--queries_in", dest="queries_in", action="store",
                    default="", help="the paperboat file with the test data")
    parser.add_argument("--learning_rate", dest="learning_rate", action="store",
                    default=0.11, help="the learning rate for ADAM optimizer", type=float)
    parser.add_argument("--batch_size", dest="batch_size", action="store",
                    default=350, help="the number of observations for each iteration during the training", type=int)
    parser.add_argument("--training_epochs", dest="training_epochs", action="store",
                    default=100, help="the number of batches to use during the training", type=int)
    parser.add_argument("--model_parameters_in", dest="model_parameters_in",\
                        action="store", default=[],\
                        help="A list of the number of neurons for each layer")
    parser.add_argument("--model_parameters_out", dest="model_parameters_out",\
                        action="store", default="",\
                        help="A json file with the extra output parameters for the model")
    parser.add_argument("--model_metrics", dest="model_metrics",\
                        action="store", default="",\
                        help="this is a json file with all the metrics,"\
                             "exported as plots or metrics")

    return parser

#metrics
def GenerateMetrics(targets, preds): 
 
  l1=0.0
  l2=0.0
  error=0.0
  sq_error=0.0
  abs_error=0.0
  count=0
  #print targets[:10]
  preds2 = map(float,preds)
  #print preds2[:10]
  for (sales, prediction) in zip(targets, preds2):
    #if (count in range(200,220)):#or(count in range(1,11)):
     #    print "sales =",sales, "||", "prediction=",prediction

    l1+=abs(sales)
    l2+=sales*sales
    err=prediction-sales
    error+=err
    sq_error+=err*err
    abs_error+=abs(err)
    count+=1
  rmse=100*np.sqrt(sq_error/l2)
  wape=100*abs_error/l1
  bias=100*error/l1
  
  return {"bias":bias, "rmse":rmse, "wape":wape}



    
