import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = T.nnet.binary_crossentropy(y_hat, y).sum(axis=-1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    ll = (y * T.log(y_hat)).sum(axis=-1)
    nll = -ll
    return nll


def NllMulInd(y, y_hat):
    """
    Multi cross-entropy
    Efficient implementation using the indices in y

    Credit assignment:
    This code is brought from: https://github.com/lisa-lab/pylearn2

    Parameters
    ----------
    .. todo::
    """
    log_prob = T.log(y_hat)
    flat_log_prob = log_prob.flatten()
    flat_y = y.flatten()
    flat_indices = flat_y + T.arange(y.shape[0]) * log_prob.shape[1]
    ll = flat_log_prob[T.cast(flat_indices, 'int64')]
    nll = -ll
    return nll


def MSE(y, y_hat, use_sum=1):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    if use_sum:
        mse = T.sum(T.sqr(y - y_hat), axis=-1)
    else:
        mse = T.mean(T.sqr(y - y_hat), axis=-1)
    return mse


def Laplace(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = T.sum(abs(y - mu) / sig + T.log(sig) + T.log(2), axis=-1)
    return nll


def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=-1)
    return nll


def GMM(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    y.name = 'y_shuffled'
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]//coeff.shape[-1],
                     coeff.shape[-1]))
    mu.name = 'mu'
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]//coeff.shape[-1],
                       coeff.shape[-1]))
    sig.name = 'sig'
    a = T.sqr(y - mu)
    a.name = 'a'
    inner = -0.5 * T.sum(a / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner.name = 'inner'
    nll = -logsumexp(T.log(coeff) + inner, axis=1)
    nll.name = 'logsum'
    return nll

def GMMdisag2(y, mu, sig, coeff, mu2, sig2, coeff2):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)
    nll = nll1 + nll2
    return nll

def GMMdisag3(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))


    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    nll = nll1 + nll2 + nll3
    return nll

def GMMdisag4(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3, mu4, sig4, coeff4):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y4 = y[:,3].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')

    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    y4.name = 'y4_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))

    mu4 = mu4.reshape((mu4.shape[0],mu4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
    sig4 = sig4.reshape((sig4.shape[0],sig4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))


    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    inner4 = -0.5 * T.sum(T.sqr(y4 - mu4) / sig4**2 + 2 * T.log(sig4) + T.log(2 * np.pi), axis=1)
    nll4 = -logsumexp(T.log(coeff4) + inner4, axis=1)

    nll = nll1 + nll2 + nll3 + nll4
    return nll

def GMMdisag5(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3, mu4, sig4, coeff4, mu5, sig5, coeff5):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y4 = y[:,3].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y5 = y[:,4].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')

    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    y4.name = 'y4_shuffled'
    y5.name = 'y5_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))

    mu4 = mu4.reshape((mu4.shape[0],mu4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
    sig4 = sig4.reshape((sig4.shape[0],sig4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))

    mu5 = mu5.reshape((mu5.shape[0],mu5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))
    sig5 = sig5.reshape((sig5.shape[0],sig5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))

    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    inner4 = -0.5 * T.sum(T.sqr(y4 - mu4) / sig4**2 + 2 * T.log(sig4) + T.log(2 * np.pi), axis=1)
    nll4 = -logsumexp(T.log(coeff4) + inner4, axis=1)

    inner5 = -0.5 * T.sum(T.sqr(y5 - mu5) / sig5**2 + 2 * T.log(sig5) + T.log(2 * np.pi), axis=1)
    nll5 = -logsumexp(T.log(coeff5) + inner5, axis=1)

    nll = nll1 + nll2 + nll3 + nll4 + nll5
    return nll

def GMMdisagMulti(y, y_dim,  mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    newY = [y[:,i].dimshuffle(0, 'x').dimshuffle(0, 1, 'x') for i in range(y_dim)]#[:,0,:]

    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = [mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1])) for i in range(y_dim)]
    sig = [sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1])) for i in range(y_dim)]

    '''
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))

    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)
    '''
    inner = [-0.5 * T.sum(T.sqr(newY[i] - mu[i]) / sig[i]**2 + 2 * T.log(sig[i]) + T.log(2 * np.pi), axis=1) for i in range(y_dim)]
    nll = [-logsumexp(T.log(coeff[i]) + inner[i], axis=1) for i in range(y_dim)]

    nll = nll.sum()
    return nll

def BiGauss(y, mu, sig, corr, binary):#x_in, theta_mu_in, theta_sig_in, corr_in, binary_in
    """
    Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    """
    mu_1 = mu[:, 0].reshape((-1, 1))
    mu_2 = mu[:, 1].reshape((-1, 1))

    sig_1 = sig[:, 0].reshape((-1, 1))
    sig_2 = sig[:, 1].reshape((-1, 1))

    y0 = y[:, 0].reshape((-1, 1))
    y1 = y[:, 1].reshape((-1, 1))
    y2 = y[:, 2].reshape((-1, 1))
    corr = corr.reshape((-1, 1))

    c_b =  T.sum(T.xlogx.xlogy0(y0, binary) +
                T.xlogx.xlogy0(1 - y0, 1 - binary), axis=1)

    inner1 =  ((0.5*T.log(1-corr**2)) +
               T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2 -
         (2. * (corr * (y1 - mu_1) * (y2 - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = - (inner1 + (inner2 * z))

    nll = -T.sum(cost ,axis=1) - c_b

    return nll


def BiGMM(y, mu, sig, coeff, corr, binary):
    """
    Bivariate Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    corr  : FullyConnected (Tanh)
    binary: FullyConnected (Sigmoid)
    """
    y = y.dimshuffle(0, 1, 'x')

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1] / coeff.shape[-1],
                     coeff.shape[-1]))

    mu_1 = mu[:, 0, :]
    mu_2 = mu[:, 1, :]

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1] / coeff.shape[-1],
                       coeff.shape[-1]))

    sig_1 = sig[:, 0, :]
    sig_2 = sig[:, 1, :]

    c_b = T.sum(T.xlogx.xlogy0(y[:, 0, :], binary) +
                T.xlogx.xlogy0(1 - y[:, 0, :], 1 - binary), axis=1)

    inner1 = (0.5 * T.log(1 - corr ** 2) +
              T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y[:, 1, :] - mu_1) / sig_1)**2 + ((y[:, 2, :] - mu_2) / sig_2)**2 -
         (2. * (corr * (y[:, 1, :] - mu_1) * (y[:, 2, :] - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = -(inner1 + (inner2 * z))

    nll = -logsumexp(T.log(coeff) + cost, axis=1) - c_b

    return nll


def KLGaussianStdGaussian(mu, sig):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and standardized Gaussian dist.

    Parameters
    ----------
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=-1)

    return kl


def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                    (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)
    else:
        kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=-1)

    return kl


def grbm_free_energy(v, W, X):
    """
    Gaussian restricted Boltzmann machine free energy

    Parameters
    ----------
    to do::
    """
    bias_term = 0.5*(((v - X[1])/X[2])**2).sum(axis=1)
    hidden_term = T.log(1 + T.exp(T.dot(v/X[2], W) + X[0])).sum(axis=1)
    FE = bias_term -hidden_term

    return FE
