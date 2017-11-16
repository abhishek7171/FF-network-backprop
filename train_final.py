import json
import random
import sys
import argparse
import matplotlib.pyplot as plt
import _pickle as cPickle
import gzip
import numpy as np

def load_data():
    
    f = gzip.open(mnist_location, 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
	
    
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vector_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vector_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e




def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float)
	parser.add_argument("--momentum", type=float)
	parser.add_argument("--num_hidden", type=int)
	parser.add_argument("--sizes", nargs='+')
	parser.add_argument("--activation", choices=["tanh", "sigmoid"])
	parser.add_argument("--loss", choices=["sq", "ce"])
	parser.add_argument("--opt", choices=["gd","momentum","nag", "adam"])
	parser.add_argument("--batch_size")
	parser.add_argument("--anneal", choices = ["True", "False"])
	parser.add_argument("--save_dir")
	parser.add_argument("--expt_dir")
	parser.add_argument("--mnist")
	args = parser.parse_args()
	global learning_rate, momentum, num_hidden, sizes, activ, loss, opt_algo, batch_size, anneal, save_dir, expt_dir, mnist_location
	learning_rate = args.lr
	momentum = args.momentum
	num_hidden = args.num_hidden
	sizes = [int(x) for x in args.sizes[0].split(',')]
	if(len(sizes) != num_hidden):
		print("\nError: num_hidden should be equal to sizes")
		exit(0)
	activ = args.activation
	loss = args.loss
	opt_algo = args.opt
	batch_size = args.batch_size
	anneal = args.anneal
	save_dir = args.save_dir
	expt_dir = args.expt_dir
	mnist_location = args.mnist


class sq(object):

    
    def fn(a, y):
        
        return 0.5*np.linalg.norm(a-y)**2
	
    
    def delta(z, a, y):
        
        return (a-y) * sigmoid_deriv(z)

class ce(object):

    
    def fn(a, y):
        
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
		
    
    def delta(z, a, y):
        
        return (a-y)

class Network(object):

    def __init__(self, sizes,cost):
        #input sizes of the layers
        self.number_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializing()
        self.cost=cost
        self.eta = 0.0
        self.count = 0.0
        self.cost_log = []

    def weight_initializing(self):
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    

    def feedforward(self, a):
        
        for b, w in zip(self.biases, self.weights):
            if activ =="tanh":
                a = np.tanh(np.dot(w, a)+b)
            else:
                a= sigmoid(np.dot(w,a)+b)
        return a
    def gradient_descent(self,anneal,training_data, epochs,batch_size, eta,
            r = 0.0,validation_data=None,testing_data=None,cost_testing = False, testing_accuracy = False, cost_training = False,training_accuracy = False,cost_validation=False,validation_accuracy=False):
            
        
        self.eta = eta
        if testing_data: 
            n_test = len(testing_data)
        n = len(training_data)
        test_cost, test_accuracy = [], []
        train_cost, train_accuracy = [], []
        validation_cost,valid_accuracy = [],[]
        ate = []
        csv = []
        cstr = []
        cste = []
        test_accuracy.append(0)
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
                
                
            for batch in batches:
                self.update_batch(batch, eta,r, len(training_data))
                    
            print ("%s epoch training" % i)
            if cost_training:
                cost = self.total_cost(training_data, r)
                train_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if training_accuracy:
                accuracy,c = self.accuracy(training_data,flag=True)
                train_accuracy.append(accuracy)
                cstr.append(c)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if cost_validation:
                cost = self.total_cost(validation_data,r,flag=True)
                validation_cost.append(cost)
                print("Cost on validation data: {}".format(cost))
            if validation_accuracy:
                accuracy,c = self.accuracy(validation_data)
                valid_accuracy.append(accuracy)
                csv.append(c)
                print ("Accuracy on validation data: {} / {}".format(
                    accuracy, n_test))
            if cost_testing:
                cost = self.total_cost(testing_data,r, flag=True)
                test_cost.append(cost)
                print ("Cost on test data: {}".format(cost))
            if testing_accuracy:
                accuracy,c = self.accuracy(testing_data,flag2=True)
                if (accuracy<test_accuracy[-1] and anneal=="True"):
                    print (self.eta)
                    self.eta=self.eta/2.0
                    print (self.eta)
                    print('halving the learning rate')					
                
                test_accuracy.append(accuracy)
                cste.append(c)
                print ("Accuracy on test data: {} / {}".format(
                    accuracy, n_test))
                print("\n error on test data is  ",c)
            ate.append(self.eta)
            m = open(expt_dir+ "/loss_validation.txt","w")
            for i in range(len(validation_cost)):
                m.write("Epoch :"+str(i)+" ,Loss: " +str(validation_cost[i])+ " ,lr: "+ str(ate[i])+"\n")
            m.close()	
            q = open(expt_dir+ "/loss_train.txt","w")
            for i in range(len(train_cost)):
                q.write("Epoch :"+str(i)+" ,Loss :" +str(train_cost[i])+ " ,lr: "+ str(eta)+"\n")
            q.close()
            p = open(expt_dir+ "/loss_test.txt","w")
            for i in range(len(test_cost)):
                p.write("Epoch :" +str(i)+" ,Loss :" +str(test_cost[i])+ " ,lr: "+ str(ate[i])+"\n")
            p.close()
     
            h = open(expt_dir+"/error_validation.txt","w")
            for i in range(len(csv)):
                h.write("Epoch :" +str(i)+" ,Error :" +str(round(csv[i],2))+ " ,lr: "+ str(ate[i])+"\n")
            h.close()
            f = open(expt_dir+ "/error_train.txt","w")
            for i in range(len(cstr)):
                f.write("Epoch :" +str(i)+" ,Error :" +str(round(cstr[i],2))+ " ,lr: "+ str(ate[i])+"\n")
            f.close()
            z = open(expt_dir+ "/error_test.txt","w")
            for i in range(len(cste)):
                z.write("Epoch :" +str(i)+" ,Error :" +str(round(cste[i],2))+" ,lr: "+ str(ate[i])+"\n")
            z.close()
        return train_cost,validation_cost,test_cost,train_accuracy,valid_accuracy,test_accuracy
           
        
    def update_batch(self,batch,eta,r, n):
        
        bias = [np.zeros(b.shape) for b in self.biases]
        weight = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            delta_bias, delta_weight = self.backpropogation(x, y)
            bias = [bi+db for bi,db in zip(bias, delta_bias)]
            weight = [we+dw for we, dw in zip(weight, delta_weight)]
        if opt_algo =="gd":
            		
            self.weights = [w-(eta/len(batch))*we
                        for w,we in zip(self.weights,weight)]
            self.biases = [b-(eta/len(batch))*bi
                       for b,bi in zip(self.biases,bias)]
        else:
            self.weights = [(1-eta*(r/n))*w-(eta/len(batch))*we
                        for w,we in zip(self.weights,weight)]
            self.biases = [b-(eta/len(batch))*bi
                       for b,bi in zip(self.biases,bias)]
    
    def backpropogation(self, x, y):
        
        bias = [np.zeros(b.shape) for b in self.biases]
        weight = [np.zeros(w.shape) for w in self.weights]
        # one feedforward
        activation = x
        activations = [x] 
        zvect = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zvect.append(z)
            if activ =="tanh":
                activation = np.tanh(z)
            else:
                activation = sigmoid(z)			
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zvect[-1], activations[-1], y)
        bias[-1] = delta
        weight[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.number_layers):
            z = zvect[-l]
            if activ =="sigmoid":
                s = sigmoid_deriv(z)
            else:
                s = 1-(np.tanh(z)**2)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * s
            bias[-l] = delta
            weight[-l] = np.dot(delta,activations[-l-1].transpose())
        return (bias,weight)
		#weights[-1]+ = del[-1]*mn
    def accuracy(self,data,flag=False,flag2=False):
        fin = []
        if flag:
            final = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            final = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        self.count = self.count + 1
        #if self.count==10:
            #print (final)
		#print (final)
        fin = np.array(final)
        #print (fin.shape)
        if flag==False and flag2==False:
            k = open("valid_predictions.txt","w")
            for x in range(fin.shape[0]):
                k.write("predicted: " + str(fin[x][0]) + " target: " + str(fin[x][1]) + "\n")
            k.close()
        if flag==False and flag2==True:
            o = open("test_predictions.txt","w")
            for x in range(fin.shape[0]):
                o.write("predicted: " + str(fin[x][0]) +"\n")
            o.close()		
        #print(np.size(data))
        temp = sum(int(x == y) for (x, y) in final)
        t = np.size(data)/2
        c = float((t-temp)/t)
        c *=100
        return temp,c
    
    def total_cost(self, data,r, flag=False):
        
        cost = 0.0
         
        
        for x, y in data:
            a = self.feedforward(x)
            if flag: y = vector_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(r/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        
        #p = open("ll.txt","w")
        #p.write(str(cost))
        #p.write("\n")
        #f.close()
        return cost

    def save(self, filename):
        
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename+ "/pred.txt", "w")
        json.dump(data, f)
        f.close()
	
    def jlt(self,anneal):
        if anneal == True:
            return True
        else:
            return False
	   	

def load(filename):
    
    f = open("pred.txt", "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vector_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    #sigmoid function.
    return 1.0/(1.0+np.exp(-z))

def sigmoid_deriv(z):
    #Derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))


def main():
    #import mnist_loader
    
	
    parse_args() 
    	
    training_data,validation_data,test_data = load_data_wrapper()	
    a = []
    a.append(784)
    i = 1
    #print (sizes)
    for i in range(num_hidden):
        a.append(sizes[i])
    a.append(10)
    print (a)	
    #for i in range(num_hidden+1):
    #print (loss)  
    if loss == "sq":	
        n = Network(a,sq)
    else:
        n = Network(a,ce)
   # print(int(learning_rate))
    #print(int(batch_size))
    #n.jlt(anneal)
    p,q,r,s,t,u= n.gradient_descent(anneal,training_data,10,int(batch_size),float(learning_rate),float(momentum),validation_data,test_data,testing_accuracy=True,cost_testing=True,cost_training=True,cost_validation=True,training_accuracy=True,validation_accuracy=True)
    s = np.array(s)
    s = s/50000.0
    t = np.array(t)
    t = t/10000.0
    #t = t/10000.0
    n.save(save_dir)
    plt.plot(p,'r',label="training loss")
    plt.plot(q,'b',label="validation loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()
    plt.plot(s,'r',label="training accuracy")
    plt.plot(t,'b',label="validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
	

    


       
     

