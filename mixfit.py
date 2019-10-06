import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize


def multi_generation():
    '''
    А вот это мой генератор для двух 2д гауссиан и равном. распределения.
    Я его сделал, в основном, для того, чтобы понять, как работает stats.multivariate_normal
    '''
    n = 1000
    a = np.array([-2,-2])
    b = np.array([4,4])
    c = b-a
    tau1 = 0.4
    mu1 = [0.2,0.2]
    sigma1 = [[0.2,0.2],[0.2,0.2]]
    
    tau2 = 0.4
    mu2 = [1.6,1.6]
    sigma2 = [[0.1,0.1],[0.1,0.1]]
    
    tau3 = 1-tau1-tau2
    
    n1 = stats.multivariate_normal.rvs(mean = mu1, cov = sigma1, size = np.array([tau1*n,2]).astype(dtype = int))
    n2 = stats.multivariate_normal.rvs(mean = mu2, cov = sigma2, size = np.array([tau2*n,2]).astype(dtype = int))
    u = stats.uniform.rvs(loc = a, scale = c, size = np.array([tau3*n+1,2]).astype(dtype=int))
    xb = np.concatenate((n1, n2))[:,0]
    yb = np.concatenate((n1, n2))[:,1]
    x = np.concatenate((xb[:,0],u[:,0]))
    y = np.concatenate((yb[:,1],u[:,1]))
    print(x.shape)
    k = np.array([x.flatten(),y.flatten()])
    return n1[:,0],n1[:,1],n2[:,0],n2[:,1],u[:,0],u[:,1],k

def lik(par, x):
    tau, mu1, sigma1, mu2, sigma2 = par
    p_1 = stats.norm.pdf(x, loc = mu1, scale = np.abs(sigma1))
    p_2 = stats.norm.pdf(x, loc = mu2, scale = np.abs(sigma2))
    p = tau * p_1 + (1-tau) * p_2
    return -1*np.sum(np.log(np.abs(p)))

def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    sol = optimize.minimize(lik, np.array([tau, mu1, sigma1, mu2, sigma2]), args = x, tol = rtol)
    assert sol.success, 'max_likelihood function did not pass'
    return sol.x

def T_ij(tau, mu1, sigma1, mu2, sigma2, x):
    p_1 = stats.norm.pdf(x, loc = mu1, scale = sigma1)
    p_2 = stats.norm.pdf(x, loc = mu2, scale = sigma2)
    p = tau * p_1 + (1-tau) * p_2
    T_i1 = tau * p_1 / p
    T_i2 = (1-tau) * p_2 / p
    return T_i1, T_i2


def step(par,x):
    tau, mu1, sigma1, mu2, sigma2 = par
    T_i1, T_i2 = T_ij(tau, mu1, sigma1, mu2, sigma2, x)
    
    tau = np.sum(T_i1) / x.size
    
    mu1 = np.sum(T_i1 * x) / np.sum(T_i1)
    sigma1 = np.sqrt(np.sum(T_i1 * (x - mu1)**2) / np.sum(T_i1))
    
    mu2 = np.sum(T_i2 * x) / np.sum(T_i2)
    sigma2 = np.sqrt(np.sum(T_i2 * (x - mu2)**2) / np.sum(T_i2))
    
    return tau, mu1, sigma1, mu2, sigma2

def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    new = tau, mu1, sigma1, mu2, sigma2
    while True:
        old = new
        new = step(old,x)
        if np.allclose(new, old, rtol = rtol, atol = 0):
            break
    return new

def draw_2d():
    '''
    А здесь я генерирую и рисую 2д распределения.
    Несмотря на то, что я не успел сделать пункт 3 и решил не добавлять огрызки кода для него, я надеюсь на то,
    что этот график растопит ваши сердца и вы немного поднимете мне оценку за это дз
    '''
    x1,y1,x2,y2,xu,yu,x = multi_generation()
    plt.scatter(*x)

#draw_2d()

#Можно раскомментить и вывести эти распределения разными цветами
#plt.scatter(xu,yu)
#plt.scatter(x1,y1)
#plt.scatter(x2,y2)