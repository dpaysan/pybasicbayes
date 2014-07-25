from __future__ import division
import numpy as np
from collections import defaultdict
from copy import deepcopy

from util.text import progprint_xrange

class ParallelTempering(object):
    def __init__(self,model,temperatures):
        temperatures = [1.] + list(sorted(temperatures))
        self.models = [deepcopy(model) for T in temperatures]
        for m,T in zip(self.models,temperatures):
            m.temperature = T

        self.swapcounts = defaultdict(int)
        self.itercount = 0

    @property
    def unit_temp_model(self):
        return self.models[0]

    @property
    def temperatures(self):
        return [m.temperature for m in self.models]

    @property
    def energies(self):
        return [m.energy for m in self.models]

    def step(self,intermediate_resamples):
        for m in self.models:
            for itr in xrange(intermediate_resamples):
                m.resample_model()

        triples = zip(self.models,self.energies,self.temperatures)
        for (M1,E1,T1), (M2,E2,T2) in zip(triples[:-1],triples[1:]):
            swap_logprob = min(0., (E1-E2)*(1./T1 - 1./T2))
            if np.log(np.random.random()) < swap_logprob:
                print 'swap at %0.3f (%0.2f,%0.2f,%0.3f, %0.3f)' % (np.exp(swap_logprob),T1,T2,E1,E2)
                M1.swap_sample_with(M2)
                self.swapcounts[(T1,T2)] += 1
            else:
                print 'no swap at delta E = %0.3f' % (E1-E2)

        self.itercount += 1

    def run(self,niter,intermediate_resamples):
        samples = []
        for itr in progprint_xrange(niter):
            self.step(intermediate_resamples)

