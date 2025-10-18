import numpy as np

def convertToNPArray(data):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
        return data
    else:
        return data
    
def relu(x):
    x = convertToNPArray(x)
    return np.maximum(0,x)

def heavyside(x):
    x=convertToNPArray(x)
    return np.where(x>0,1,0)

def sigmoid(x):
    x=convertToNPArray(x)
    return 1/(1+np.exp(-x))

def tanh(x):
    x=convertToNPArray(x)
    return np.tanh(x)

def PReLU(x,a):
    x=convertToNPArray(x)
    return np.where(x>0,x,a*x)

def LeakyReLU(x,alpha=0.01):
    x=convertToNPArray(x)
    return PReLU(x,alpha)

def ParameterisedSigmoid(x,a=1):
    x=convertToNPArray(x)
    return 1/(1+np.exp(-1*(x/a)))

def softsign(x):
    x=convertToNPArray(x)
    return x/(1+np.abs(x))

def softplus(x):
    x=convertToNPArray(x)
    return np.log(1+np.exp(x))

def SiLU(x):
    x=convertToNPArray(x)
    return x*sigmoid(x)

def ELiSH(x):
    x=convertToNPArray(x)
    return np.where(x>0,x,sigmoid(x)*(np.exp(x)-1))

def GeLU(x):
    x=convertToNPArray(x)
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*(x**3))))






    