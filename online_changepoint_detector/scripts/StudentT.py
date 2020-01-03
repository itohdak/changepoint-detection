from __future__ import division
import numpy as np
from scipy import stats
from scipy.special import gammaln

class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])
	self.curr_mean = self.mu0
	self.curr_cov = self.beta0
	self.saved_mean = []
	self.saved_cov = []

    def pdf(self, data):
        return stats.t.pdf(x=data,
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

    def curr_theta(self):
	self.curr_mean = self.mu[-2]
	self.curr_cov = self.beta[-2] * 2*(self.kappa[-2]+1) / (self.alpha[-2] * self.kappa[-2])

    def save_theta(self):
	if np.size(self.saved_mean):
	   self.saved_mean = np.concatenate((self.saved_mean, [self.curr_mean]))
	else:
	   self.saved_mean = [self.curr_mean]
	if np.size(self.saved_cov):
	   self.saved_cov = np.concatenate((self.saved_cov, [self.curr_cov]))
	else:
	   self.saved_cov = [self.curr_cov]

    def reset_theta(self, t):
	self.mu = self.mu[0:t+1]
        self.kappa = self.kappa[0:t+1]
        self.alpha = self.alpha[0:t+1]
        self.beta = self.beta[0:t+1]

    def retrieve_theta(self):
	return self.saved_mean, self.saved_cov
