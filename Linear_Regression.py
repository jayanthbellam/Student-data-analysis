# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 08:40:31 2020

@author: jayan
"""
import numpy as np
def convert_to_array(data):
    for i in range(len(data)):
        data[i]=data[i].split(";")
    return data[1:]
def convert_to_int(data):
    for i in data:
        i[0]=1 if i[0]=='"GP"' else 0
        i[1]=1 if i[1]=='"F"' else 0
        i[2]=int(i[2])
        i[3]=1 if i[3]=='"U"' else 0
        i[4]=1 if i[4]=='"LE3"' else 0
        i[5]=1 if i[5]=='"T"' else 0
        i[6]=int(i[6])
        i[7]=int(i[7])
        i[8]=check_job(i[8])
        i[9]=check_job(i[9])
        i[14]=int(i[14])
        i[12]=int(i[12])
        i[13]=int(i[13])
        i[15]=check_yn(i[15])
        i[16]=check_yn(i[16])
        i[17]=check_yn(i[17])
        i[18]=check_yn(i[18])
        i[19]=check_yn(i[19])
        i[20]=check_yn(i[20])
        i[21]=check_yn(i[21])
        i[22]=check_yn(i[22])
        i[30]=int(i[30][1:-1])
        i[31]=int(i[31][1:-1])
        i[11]=1 if i[11]=='"mother"' else 0
        if i[10]=='"home"':
            i[10]=0
        if i[10]=='"reputation"':
            i[10]=1
        if i[10]=='"course"':
            i[10]=2
        else:
            i[10]=3
    return data
def check_job(job):
    if job == '"teacher"':
        return 4
    if job == '"health"':
        return 3
    if job == '"services"':
        return 2
    if job == '"at_home"':
        return 1
    else:
        return 0
def check_yn(ans):
    if ans=='"yes"':
        return 1
    else:
        return 0
    testing_set=data[:,:int(9*0.1*len(data))]
    training_set = data[:,int(9*0.1*len(data)):]
    return training_set,testing_set
def inputoutput(inset):
    input_set=inset[:30,:]
    outset=inset[30:31,:]//10
    return input_set, outset
def initialize(shape_input,shape_output):
    w=np.random.rand(shape_input[0],shape_output[0])
    b=0
    return w,b
def sigmoid(x):
    return 1/(1+np.exp(-x))
def propagate(w,b,X,Y):
        
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T,X) +b)    # compute activation
    cost =-1 * 1/m *( np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1-A).T)     )              # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * (np.sum((A-Y),axis=1,dtype=float))[0]
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads,cost=propagate(w,b,X,Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###
        
        
        # Print the cost every 100 training iterations
        if print_cost and i % 10_000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads
file=open('student-por.csv')
data=file.readlines()
file.close()
data=convert_to_array(data)
data=data
data=np.array(convert_to_int(data),dtype=np.int32).T
training_set=data
X,Y=inputoutput(training_set)
del data,training_set,file
w,b=initialize(X.shape,Y.shape)
params,grads=optimize(w, b, X, Y,1_000_000,0.001,True)
w=params["w"]
b=params["b"]
y_predict=sigmoid(np.dot(w.T,X)+b)//0.5
Accuracy=Y+y_predict
count=0
for i in range(len(Accuracy[0])):
    if Accuracy[0][i]==2.0:
        count+=1
    if Accuracy[0][i]==0.0:
        count+=1
print(count/len(Accuracy[0]))
del params,grads,X,Y
file=open('student-mat.csv')
data=file.readlines()
file.close()
data=convert_to_array(data)
data=data
data=np.array(convert_to_int(data),dtype=np.int32).T
training_set=data
X,Y=inputoutput(training_set)
y_predict=sigmoid(np.dot(w.T,X)+b)//0.5
Accuracy=Y+y_predict
count=0
for i in range(len(Accuracy[0])):
    if Accuracy[0][i]==2.0:
        count+=1
    if Accuracy[0][i]==0.0:
        count+=1
print(count/len(Accuracy[0]))