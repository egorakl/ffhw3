import mixfit
import numpy as np
from scipy import stats

def data_generation(tau,mu1,sigma1,mu2,sigma2):
    n = 10000

    x_n1 = stats.norm.rvs(loc = mu1, scale = sigma1, size = int(tau*n))
    x_n2 = stats.norm.rvs(loc = mu2, scale = sigma2, size = int((1-tau)*n))
    x = np.concatenate((x_n1, x_n2))
    return x

x0 = [0.5,0.2,0.2,0.6,0.1] #tau, mu1, sigma1, mu2, sigma2
data = data_generation(*x0)

ml = mixfit.max_likelihood(data,0.3,0.5,0.5,0.5,0.3)
em = mixfit.em_double_gauss(data,0.3,0.5,0.5,0.5,0.3)

np.testing.assert_allclose(x0, ml, rtol = 1e-1)
np.testing.assert_allclose(x0, em, rtol = 1e-1)

print('parameters: ',x0)
print('max likelihood: ',ml)
print('em algorithm: ',em)