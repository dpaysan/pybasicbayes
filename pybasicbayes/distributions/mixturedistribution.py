from __future__ import division
from builtins import zip
from builtins import map
from builtins import range

import numpy as np
from warnings import warn
import scipy.stats as stats
import scipy.special as special
import copy

from pybasicbayes.abstractions import GibbsSampling, MeanField, MeanFieldSVI, \
    MaxLikelihood, MAP

from pybasicbayes.distributions import *


class DistributionMixture(GibbsSampling, MeanField, MeanFieldSVI,
                          MaxLikelihood, MAP):
    """
    This class represents a mixture of a different types of distributions implementing the interfaces
    defined in pyhsmm.abstractions (see pybasicbayes.distribution for examples)

    Parameters:
        distv:          list of distributions forming the mxiture
        dist_obs_map:   list of length of the observations, mapping the obsevations to the respective
                        distribution in distv by the respective index e.g. [1,1,0,2] means the first
                        variables of each observation will be modeled with the second distribution in distv,
                        the third variable will be modeled with the first distribution in distv and so on.

    """

    def __init__(self, distv, dist_obs_map):

        self.distv = distv
        self.dist_obs_map = dist_obs_map

    def rvs(self, size=None):
        return np.hstack(
            [self.distv[idx].rvs(size) for idx in self.dist_obs_map])

    def log_likelihood(self, x):
        out = np.zeros_like(x.shape[0], dtype=np.double)
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            out += self.distv[idx].log_likelihood(x=x, mask=mask)
        return out

    ## Gibbs sampling

    def resample(self, data=[]):
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            self.distv[idx] = self.distv[idx].resample(data, mask)
        return self

    def _get_statistics(self, data):
        distv_statistics = []
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            dist = self.distv[idx]
            if isinstance(dist, Multinomial):
                distv_statistics[idx] = dist._get_statistics(data=data,
                                                             K=dist.K,
                                                             mask=mask)
            elif isinstance(dist, Gaussian):
                distv_statistics[idx] = dist._get_statistics(data=data,
                                                             D=dist.D,
                                                             mask=mask)
        return distv_statistics

    def _get_weighted_statistics(self, data, weights):
        distv_weighted_statistics = []
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            dist = self.distv[idx]
            if isinstance(dist, Categorical):
                distv_weighted_statistics[idx] = dist._get_weighted_statistics(
                    data=data, weights=weights)
            elif isinstance(dist, Gaussian):
                distv_weighted_statistics[idx] = dist._get_weighted_statistics(
                    data=data, weights=weights, D=dist.D)
        return distv_weighted_statistics

    ### Mean Field

    def meanfieldupdate(self, data, weights):
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            self.distv[idx] = self.distv[idx].meanfieldupdate(data=data,
                                                              weights=weights,
                                                              mask=mask)
        return self

    def get_vlb(self):
        distv_vlbs = []
        for idx in range(len(self.distv)):
            distv_vlbs[idx] = self.distv[idx].get_vlb()
        return distv_vlbs

    def expected_log_likelihood(self, x):
        out = np.zeros_like(x.shape[0], dtype=np.double)
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            out += self.distv[idx].expected_log_likelihood(x=x, mask=mask)
        return out

    ### Mean Field SGD

    def meanfield_sgdstep(self, data, weights, prob, stepsize):
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            self.distv[idx] = self.distv[idx].meanfieldupdate_sgd(data=data,
                                                                  weights=weights,
                                                                  prob=prob,
                                                                  stepsize=stepsize,
                                                                  mask=mask)
        return self

    def max_likelihood(self, data, weights=None):
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            self.distv[idx] = self.distv[idx].max_likelihood(data=data,
                                                             weights=weights,
                                                             mask=mask)
        return self

    def MAP(self, data, weights=None):
        for idx in np.unique(self.dist_obs_map):
            mask = self.dist_obs_map == idx
            self.distv[idx] = self.distv[idx].MAP(data=data,
                                                  weights=weights,
                                                  mask=mask)
        return self
