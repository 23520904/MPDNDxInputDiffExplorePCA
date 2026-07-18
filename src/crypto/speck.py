import numpy as np
from os import urandom

MASK_VAL = 2 ** WORD_SIZE() - 1

def WORD_SIZE():
    return (16)

def ALPHA():
    return (7)

def BETA():
    return (2)

def left_round(value, shiftBits):
	t1 = (value >> (WORD_SIZE() - shiftBits)) ^ (value << shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2

def right_round(value, shiftBits):
	t1 = (value << (WORD_SIZE() - shiftBits)) ^ (value >> shiftBits)
	t2 = ((2 ** WORD_SIZE()) - 1)
	return t1 & t2


def enc_one_round(p,k):
    c0, c1 = p[0],p[1]
    c0 = right_round(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = left_round(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)

def expand_keys(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)

def encryption(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

def convert_to_binary(arr,s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups,len(arr[0])),dtype=np.uint8) 
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)


def make_train_data(n, nr, a=((0x0040,0x0), (0x0,0x8000), (0x0060,0x0)), b=((0x0020,0x0),(0x0040,0x8000),(0x0010,0x2000)), s_groups=1):
    
    Y = np.frombuffer(urandom(n),dtype = np.uint8)
    Y = Y&1
    num_rand_samples = np.sum(Y==0)

    key = np.frombuffer(urandom(8*n), dtype = np.uint16).reshape(4,-1)

    keys = expand_keys(key, nr)

    X = []

    for i in range(s_groups):
        plain1_1 = np.frombuffer(urandom(2*n),dtype=np.uint16)
        plain1_2 = np.frombuffer(urandom(2*n),dtype=np.uint16)

        plain2_1 = np.where(Y == 0, plain1_1 ^ a[0][0], plain1_1 ^ b[0][0])
        plain2_2 = np.where(Y == 0, plain1_2 ^ a[0][1], plain1_2 ^ b[0][1])
        
        plain3_1 = np.where(Y == 0, plain1_1 ^ a[1][0], plain1_1 ^ b[1][0])
        plain3_2 = np.where(Y == 0, plain1_2 ^ a[1][1], plain1_2 ^ b[1][1])
       
        plain4_1 = np.where(Y == 0, plain1_1 ^ a[2][0], plain1_1 ^ b[2][0])
        plain4_2 = np.where(Y == 0, plain1_2 ^ a[2][1], plain1_2 ^ b[2][1])

        cipher1_1, cipher1_2 = encryption((plain1_1, plain1_2), keys)
        cipher2_1, cipher2_2 = encryption((plain2_1, plain2_2), keys)
        cipher3_1, cipher3_2 = encryption((plain3_1, plain3_2), keys)
        cipher4_1, cipher4_2 = encryption((plain4_1, plain4_2), keys)

        X.append(cipher1_1)
        X.append(cipher1_2)
        X.append(cipher2_1)
        X.append(cipher2_2)
        X.append(cipher3_1)
        X.append(cipher3_2)
        X.append(cipher4_1)
        X.append(cipher4_2)

    XT = convert_to_binary(X,s_groups=s_groups)

    return XT,Y