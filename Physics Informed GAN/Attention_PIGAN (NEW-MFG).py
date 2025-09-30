import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import os
import pandas as pd
import os
save_dir = "//home//naman//Robust_MFG//Model_Checkpoint"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0, 1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("Set memory growth to True.")
#     except RuntimeError as e:
#         print(e)
seed = 1234
np.random.seed(seed)
# tf.set_random_seed(seed)


# Traffic Data


datarho = pd.read_csv('//home//naman//Robust_MFG//1._Decreasing_term_cost//LWRMFG2_density.csv').values
dataC = pd.read_csv('//home//naman//Robust_MFG//1._Decreasing_term_cost//LWRMFG2_cost.csv').values #Terminal Cost C_T(x, y) = -y/10
# datav = pd.read_csv('//home//naman//Robust_MFG//MFG_LWR_velocityv.csv').values
rho = datarho.reshape(420, 60, 60)
C = dataC.reshape(420, 60, 60)
# C = 
# u = datau.reshape(320, 80, 80)
# v = datav.reshape(320, 80, 80)
sigma = 0.001
u_max = 1.02
rho_max = 1.13
v_max = 0.15
k = 100
L = 1
b = 0.3
T_max = 2
m, n, o = rho.shape
x = np.linspace(0, L, n)
t = np.linspace(0, T_max, m)
y = np.linspace(-b, b, o)

N_obs = n*o+2*m*n+2*m*o
N_auxHJB = 75000
N_auxFPK = 75000
X, T, Y = np.meshgrid(x, t, y)

inputX = np.hstack([T.flatten()[:, None], X.flatten()[:, None], Y.flatten()[:, None]])
rho_star = rho.flatten()[:, None]
# pd.DataFrame(inputX).to_csv('//home//naman/Robust_MFG//datainput.csv', index = False)
C_star = C.flatten()[:, None]

t0 = np.hstack([T[0:1, :, :].reshape(n*o, 1), X[0:1, :, :].reshape(n*o, 1), Y[0:1, :, :].reshape(n*o, 1)]) #Initial condition data (0, x, y)
x0 = np.hstack([T[:, 0:1, :].reshape(m*o, 1), X[:, 0:1, :].reshape(m*o, 1), Y[:, 0:1, :].reshape(m*o, 1)]) #Boundary condition data (t, 0, x) 
xL = np.hstack([T[:, -1:, :].reshape(m*o, 1), X[:, -1:, :].reshape(m*o, 1), Y[:, -1:, :].reshape(m*o, 1)]) #Boundary condition data (t, L, y)
y0 = np.hstack([T[:, :, 0:1].reshape(m*o, 1), X[:, :, 0:1].reshape(m*o, 1), Y[:, :, 0:1].reshape(m*o, 1)]) #Boundary condition data (t, x, -b) 
yb = np.hstack([T[:, :, -1:].reshape(m*o, 1), X[:, :, -1:].reshape(m*o, 1), Y[:, :, -1:].reshape(m*o, 1)]) #Boundary condition data (t, x, b)
tT = np.hstack([T[-1:, :, :].reshape(n*o, 1), X[-1:, :, :].reshape(n*o, 1), Y[-1:, :, :].reshape(n*o, 1)]) #Initial condition data (0, x, y)

rhot0 = rho[0:1, :, :].reshape(n*o, 1) #rho(0, x, y)
rhox0 = rho[:, 0:1, :].reshape(m*o, 1) #rho(t, 0, y)
rhoxL = rho[:, -1:, :].reshape(m*o, 1) #rho(t, L, y)
rhoy0 = rho[:, :, 0:1].reshape(m*n, 1) #rho(t, x, -b)
rhoyb = rho[:, :, -1:].reshape(m*n, 1) #rho(t, x, b)

CtT = C[-1:, :, :].reshape(n*o, 1) #C(T, x, y)
Cx0 = C[:, 0:1, :].reshape(m*o, 1) #C(t, 0, y)
CxL = C[:, -1:, :].reshape(m*o, 1) #C(t, L, y)
Cy0 = C[:, :, 0:1].reshape(m*n, 1) #C(t, x, -b)
Cyb = C[:, :, -1:].reshape(m*n, 1) #C(t, x, b)

X_trainFPK = np.vstack([t0, x0, xL, y0, yb])
X_trainHJB = np.vstack([tT, x0, xL, y0, yb])
rho_train = np.vstack([rhot0, rhox0, rhoxL, rhoy0, rhoyb])
X_trainbound = np.vstack([y0, yb])
C_train = np.vstack([CtT, Cx0, CxL, Cy0, Cyb])
# C_train = C_train[idx]\
lb = inputX.min(0)
ub = inputX.max(0)
auxHJB = lb +(ub-lb)*lhs(3, N_auxHJB)
auxFPK = lb+(ub-lb)*lhs(3, N_auxFPK)
# print(C_train)
# input('Press Enter')
class robust_MFG_Attention_PIGAN:
    def __init__(self, trainFPK, trainHJB, X_trainbound, rho, C, auxFPK, auxHJB, lb, ub, layersG, layersD):
        self.lb = lb
        self.ub = ub
        
        self.tFPK = trainFPK[:, 0:1]
        self.xFPK = trainFPK[:, 1:2]
        self.yFPK = trainFPK[:, 2:3]
        self.tHJB = trainHJB[:, 0:1]
        self.xHJB = trainHJB[:, 1:2]
        self.yHJB = trainHJB[:, 2:3]
        self.tb = X_trainbound[:, 0:1]
        self.xb = X_trainbound[:, 1:2]
        self.yb = X_trainbound[:, 2:3]
        self.Aht = auxHJB[:, 0:1]
        self.Ahx = auxHJB[:, 1:2]
        self.Ahy = auxHJB[:, 2:3]
        self.Aft = auxFPK[:, 0:1]
        self.Afx = auxFPK[:, 1:2]
        self.Afy = auxFPK[:, 2:3]
        self.C = C
        self.rho = rho
        self.Wg, self.Bg = self.initialise_networks(layersG)#Generator parameters
        self.Wd, self.Bd = self.initialise_networks(layersD)#discriminator parameters
        self.Wug, self.Bug = self.initialise_networks([3, 10])
        self.Wvg, self.Bvg = self.initialise_networks([3, 10])
        self.Wud, self.Bud = self.initialise_networks([3, 10])
        self.Wvd, self.Bvd = self.initialise_networks([3, 10])

        self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True))
        self.xFp = tf.placeholder(tf.float32, shape = [None, self.xFPK.shape[1]])
        self.tFp = tf.placeholder(tf.float32, shape = [None, self.tFPK.shape[1]])
        self.yFp = tf.placeholder(tf.float32, shape = [None, self.yFPK.shape[1]])
        self.xHp = tf.placeholder(tf.float32, shape = [None, self.xHJB.shape[1]])
        self.tHp = tf.placeholder(tf.float32, shape = [None, self.tHJB.shape[1]])
        self.yHp = tf.placeholder(tf.float32, shape = [None, self.yHJB.shape[1]])       
        self.tbp = tf.placeholder(tf.float32, shape = [None, self.tb.shape[1]])
        self.xbp = tf.placeholder(tf.float32, shape = [None, self.xb.shape[1]])
        self.ybp = tf.placeholder(tf.float32, shape = [None, self.yb.shape[1]])
        self.rhop = tf.placeholder(tf.float32, shape = [None, self.rho.shape[1]])
        self.Cp = tf.placeholder(tf.float32, shape = [None, self.C.shape[1]])
        self.Ahxp = tf.placeholder(tf.float32, shape = [None, self.Ahx.shape[1]])
        self.Ahtp = tf.placeholder(tf.float32, shape = [None, self.Aht.shape[1]])
        self.Ahyp = tf.placeholder(tf.float32, shape = [None, self.Ahy.shape[1]])
        self.Afxp = tf.placeholder(tf.float32, shape = [None, self.Afx.shape[1]])
        self.Aftp = tf.placeholder(tf.float32, shape = [None, self.Aft.shape[1]])
        self.Afyp = tf.placeholder(tf.float32, shape = [None, self.Afy.shape[1]])

        self.gen = self.gene(self.xFp, self.yFp, self.tFp)
        self.dis = self.disc(self.xHp, self.yHp, self.tHp)
        self.hatrho = self.gen[:, 0:1]
        self.hatC = self.dis[:, 0:1]
        self.FPKres = self.FPK(self.Afxp, self.Afyp, self.Aftp)[0]
        self.HJBres = self.HJBI(self.Ahxp, self.Ahyp, self.Ahtp)[0]

        self.neumannrho = self.FPK(self.xbp, self.ybp, self.tbp)[1]
        self.neumannC = self.HJBI(self.xbp, self.ybp, self.tbp)[1]
        self.g_loss = 25*tf.reduce_mean(tf.square(self.hatrho-self.rhop))+tf.reduce_mean(tf.square(self.FPKres))+tf.reduce_mean(tf.square(self.neumannrho))
        self.d_loss = 5*tf.reduce_mean(tf.square(self.hatC-self.Cp))+tf.reduce_mean(tf.square(self.HJBres))+tf.reduce_mean(tf.square(self.neumannC))
        
        self.d_optimiser = tf.contrib.opt.ScipyOptimizerInterface(self.d_loss, method = 'L-BFGS-B', options = {'maxiter': 1500,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' :1.0 * np.finfo(float).eps})
        self.g_optimiser = tf.contrib.opt.ScipyOptimizerInterface(self.g_loss, method = 'L-BFGS-B', options = {'maxiter': 1500,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' :1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def update_data(self, data1, data2):
        self.Aft = data1[:, 0:1]
        self.Afx = data1[:, 1:2]
        self.Afy = data1[:, 2:3]
        self.Aht = data2[:, 0:1]
        self.Ahx = data2[:, 1:2]
        self.Ahy = data2[:, 2:3] 
        
    def initialise_networks(self, layers):
        Weights = []
        Biases = []
        for l in range(len(layers)-1):
            std = (2/(layers[l]+layers[l+1]))**0.5
            W = tf.Variable(tf.truncated_normal([layers[l], layers[l + 1]], mean=0, stddev = std), dtype=tf.float32)
            B = tf.Variable(tf.zeros([layers[l + 1]]), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)
        return Weights, Biases
        
    def generator(self, X, Wg, Bg, Wug, Bug, Wvg, Bvg):
        L = len(Wg)+1
        Wu, bu = Wug[0], Bug[0]
        Wv, bv = Wvg[0], Bvg[0]
        A = 2*(X-self.lb)/(self.ub-self.lb)-1
        U = tf.add(tf.matmul(A, Wu), bu)
        V = tf.add(tf.matmul(A, Wv), bv)
        for l in range(L-2):
            W = Wg[l]
            b = Bg[l]
            Z = tf.add(tf.matmul(A, W), b)
            A = tf.tanh(Z)
            A = (1-A)*U+A*V
        W = Wg[-1]
        b = Bg[-1]
        Z = tf.add(tf.matmul(A, W), b)
        Y = tf.tanh(Z)
        return Y

    def discriminator(self, X, Wd, Bd, Wud, Bud, Wvd, Bvd):
        L = len(Wd)+1
        Wu, bu = Wud[0], Bud[0]
        Wv, bv = Wvd[0], Bvd[0] 
        A = 2*(X-self.lb)/(self.ub-self.lb)-1
        U = tf.add(tf.matmul(A, Wu), bu)
        V = tf.add(tf.matmul(A, Wv), bv)
        for l in range(L-2):
            W = Wd[l]
            b = Bd[l]
            Z = tf.add(tf.matmul(A, W), b)
            A = tf.tanh(Z)
            A = (1-A)*U+A*V
        W = Wd[-1] 
        b = Bd[-1]
        Z = tf.add(tf.matmul(A, W), b)
        Y = Z
        return Y

    def gene(self, x, y, t):
        Ygen = self.generator(tf.concat([t, x, y], 1) ,self.Wg, self.Bg, self.Wug, self.Bug, self.Wvg, self.Bvg)
        return Ygen
    def disc(self, x, y, t):
        Ydis = self.discriminator(tf.concat([t, x, y], 1), self.Wd, self.Bd, self.Wud, self.Bud, self.Wvd, self.Bvd)
        return Ydis

    def FPK(self, x, y, t):
        NN1 = self.gene(x, y, t)
        NN2 = self.disc(x, y, t)
        rho = NN1[:, 0:1]
        C = NN2[:, 0:1]
        rho_t = tf.gradients(rho, t)[0]
        C_x = tf.gradients(C, x)[0]
        C_y = tf.gradients(C, y)[0]
        U_eq = u_max*(1-rho/rho_max)
        V_eq = v_max*(tf.exp(-(b+y)/2)-tf.exp(-(b-y)/2))*(1-rho/rho_max)
        u = tf.clip_by_value(U_eq-C_x, 0, u_max)
        v = tf.clip_by_value((V_eq-C_y)/(1+rho/(v_max*rho_max)), -v_max, v_max)
        d1 = 1/(2*k**2)*C_x
        d2 = 1/(2*k**2)*C_y
        rhoud1_x = tf.gradients(rho*(u+d1), x)[0]
        rhovd2_y = tf.gradients(rho*(v+d2), y)[0]
        rho_y = tf.gradients(rho, y)[0]
        rho_yy = tf.gradients(rho_y, y)[0]
        f = rho_t+rhoud1_x+rhovd2_y-sigma**2/2*rho_yy
        return f, rho_y
    def HJBI(self, x, y, t):
        NN1 = self.gene(x, y, t)
        NN2 = self.disc(x, y, t)
        rho = NN1[:, 0:1]
        C = NN2[:, 0:1]
        C_t = tf.gradients(C, t)[0]
        C_x = tf.gradients(C, x)[0]
        C_y = tf.gradients(C, y)[0]
        C_yy = tf.gradients(C_y, y)[0]
        U_eq = u_max*(1-rho/rho_max)
        V_eq = v_max*(tf.exp(-(b+y)/2)-tf.exp(-(b-y)/2))*(1-rho/rho_max)
        u = tf.clip_by_value(U_eq-C_x, 0, u_max)
        v = tf.clip_by_value((V_eq-C_y)/(1+rho/(v_max*rho_max)), -v_max, v_max)
        d1 = 1/(2*k**2)*C_x
        d2 = 1/(2*k**2)*C_y
        f = 1/k**2*(C_t+0.5*(U_eq-u)**2+0.5*(V_eq-v)**2+0.5*v**2*rho/(v_max*rho_max)+(u+d1)*C_x+(v+d2)*C_y+sigma**2/2*C_yy)-(d1**2+d2**2)
        return f, C_y
    def callback(self, loss):
        print('Loss:', loss)
    def train_GAN_lbfgsb(self, epochs):
        feed_dict1 = {self.xHp: self.xHJB, self.tHp: self.tHJB, self.yHp: self.yHJB, self.Ahxp: self.Ahx, self.Ahtp: self.Aht, self.Ahyp: self.Ahy, self.Cp: self.C, self.tbp: self.tb, self.xbp: self.xb, self.ybp: self.yb}
        feed_dict2 = {self.xFp: self.xFPK, self.tFp: self.tFPK, self.yFp: self.yFPK, self.Afxp: self.Afx, self.Aftp: self.Aft, self.Afyp: self.Afy, self.rhop: self.rho, self.tbp: self.tb, self.xbp: self.xb, self.ybp: self.yb}
        for epochs in range(epochs):
            print('Training the HJBI:')
            self.d_optimiser.minimize(self.sess, feed_dict = feed_dict1, fetches = [self.d_loss], loss_callback = self.callback)
            print('Training the FPK:')
            self.g_optimiser.minimize(self.sess, feed_dict = feed_dict2, fetches = [self.g_loss], loss_callback = self.callback)

    def predict(self, inputX):
        rho_star = self.sess.run(self.gen, {self.tFp: inputX[:, 0:1], self.xFp: inputX[:, 1:2], self.yFp: inputX[:, 2:3]})
        C_star = self.sess.run(self.dis, {self.tHp: inputX[:, 0:1], self.xHp: inputX[:, 1:2], self.yHp: inputX[:, 2:3]})
        return rho_star, C_star

    # def save_model(self, path='//home//naman//Robust_MFG//Attention_PIGAN.ckpt'):
    #     save_path = self.saver.save(self.sess, path)
    #     print("Model saved in path:", save_path)
    # def load_model(self, path='//home//naman//Robust_MFG//Attention_PIGAN.ckpt'):
    #     self.saver.restore(self.sess, path)
    #     print("Model restored from path:", path)

layersg = [3, 10, 10, 10, 10, 10, 10, 10, 1]
layersd = [3, 10, 10, 10, 10, 10, 10, 10, 1]
# print(layersd)
# input('Press Enter')
model = robust_MFG_Attention_PIGAN(X_trainFPK, X_trainHJB, X_trainbound, rho_train, C_train, auxFPK, auxHJB, lb, ub, layersg, layersd)
initial_time = time.time()
model.train_GAN_lbfgsb(epochs = 10)
print('Starting the adaptive resampling...')
K = 2
c = 0
lb1 = np.array([0, 0, -0.3])
ub1 = np.array([2, 1, 0.3])
for i in range(4):
    test_aux = lb1+(ub1-lb1)*lhs(3, N_auxFPK)
    Y = np.abs(model.predict(test_aux)[0][:, 0:1])
    err_eq = np.power(Y, K)/np.power(Y, K).mean()+c
    err_eq_normalised = (err_eq/sum(err_eq))[:, 0]
    X_ids = np.random.choice(a = len(test_aux), size = 5000, replace = False, p = err_eq_normalised)
    auxFPK1 = test_aux[X_ids]
    auxFPK = np.vstack([auxFPK, auxFPK1])
    print(auxFPK.shape)
    model.update_data(auxFPK, auxHJB)
    model.train_GAN_lbfgsb(epochs = 5)

elapsed = time.time() - initial_time
Y_pred, C_pred = model.predict(inputX)
hatrho = Y_pred[:, 0:1]
hatC = C_pred[:, 0:1]
print('Time elapsed:', time.time()-initial_time)
print(hatC)
print(hatrho)
pd.DataFrame(hatrho).to_csv('//home//naman//Robust_MFG//MFGLWR2densityAttention.csv', index = False)
# pd.DataFrame(hatu).to_csv('//home//naman//Robust_MFG//LWRvelocityu.csv', index = False)
# pd.DataFrame(hatv).to_csv('//home//naman//Robust_MFG//LWRvelocityv.csv', index = False)
print('Processing Complete...')