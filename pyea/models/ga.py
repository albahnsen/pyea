"""
This module include the Genetic Algorithms method. 
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

import numpy as np
from numpy.core.defchararray import add
import pandas as pd
from scipy.stats import norm


class GeneticAlgorithmOptimizer:
    """Genetic Algorithms method. 
       Works with either binary or continuous functions

    Parameters
    ----------
    fitness_function : object
        The function to measure the performance of each chromosome.
        Must accept a vector of chromosomes and return a vector of cost.
        By default it is assumed that it is a minimization problem.
        
    n_parameters : int
        Number of parameters to estimate.
    
    iters : int, optional (default=10)
        Number of iterations
        
    type : string, optional (default="cont")
        Type of problem, either continuous or binary ("cont" or "binary")

    n_chromosomes : int, optional (default=10)
        Number of chromosomes.
        
    per_mutations : float, optional (default=0.2)
        Percentage of mutations.
            
    n_elite : int, optional (default=2)
        Number of elite chromosomes.

    fargs : tuple, optional (default=())
        Additional parameters to be send to the fitness_function

    range_ : tuple of floats, optional (default = (-1, 1))
        Only for continuous problems. The range of the search space.

    verbose : int, optional (default=0)
        Controls the verbosity of the GA process.


    Attributes
    ----------
    `hist_` : pandas.DataFrame of shape = [n_chromosomes * iters, columns=['iter', 'x0', 'x1', .., 'xk', 'cost', 'orig']]
        History of the optimization process

    References
    ----------
    .. [1] R. Haupt, S. Haupt, A.Correa, "Practical genetic algorithms (Second Edi.)",
           New Jersey: John Wiley & Sons, Inc. (2004).

    .. [2] S. Marslan, "Machine Learning: An Algorithmic Perspective",
           New Jersey, USA: CRC Press. (2009).

    Examples
    --------
    >>> import numpy as np
    >>> from pyea.models import GeneticAlgorithmOptimizer
    >>> from pyea.functions import func_rosenbrock
    >>> f = GeneticAlgorithmOptimizer(func_rosenbrock, n_parameters=2, iters=10, verbose=1)
    >>> f.fit()
    >>> # Best per iter
    >>> f.hist_[['iter', 'cost']].groupby('iter').aggregate(np.min)
    """

    def __init__(self,  
                 fitness_function,
                 n_parameters,
                 iters=10, 
                 type_='cont', 
                 n_chromosomes=10, 
                 per_mutations=0.2, 
                 n_elite=2, 
                 fargs=(),
                 range_=(-1, 1),
                 verbose=0):

        self.n_parameters = n_parameters
        self.n_chromosomes = n_chromosomes
        self.per_mutations = per_mutations
        self.iters = iters
        self.fitness_function = fitness_function
        self.fargs = fargs
        self.n_elite = n_elite
        self.type_ = type_
        self.range_ = range_
        self.verbose = verbose

    def _random(self, n):
        #TODO: Add random state
        if self.type_ == 'binary':
            temp_ = np.random.binomial(1, 0.5, size=(n, self.n_parameters)).astype(np.int)
        else:
            temp_ = (self.range_[1] - self.range_[0]) * np.random.rand(n, self.n_parameters) + self.range_[0]
        return temp_

    def fit(self):

        # Results
        _cols_x = ['x%d' % i for i in range(self.n_parameters)]
        self.hist_ = pd.DataFrame(index=range((self.iters + 1) * self.n_chromosomes),
                                  columns=['iter', ] + _cols_x + ['cost', 'orig', ])
        self.hist_[['orig', ]] = '-1'

        #Initial random population
        self.pop_ = self._random(self.n_chromosomes)
        self.cost_ = self.fitness_function(self.pop_, *self.fargs)

        filter_iter = range(0, self.n_chromosomes)
        self.hist_.loc[filter_iter, 'iter'] = 0
        self.hist_.loc[filter_iter, 'cost'] = self.cost_
        self.hist_.loc[filter_iter, _cols_x] = self.pop_

        for i in range(self.iters):

            if self.verbose > 0:
                print 'Iteration ' + str(i) + ' of ' + str(self.iters)

            orig = np.empty(self.n_chromosomes, dtype='S10')
            cost_sort = np.argsort(self.cost_)

            #Elitims
            new_pop = np.empty_like(self.pop_)
            new_pop[0:self.n_elite] = self.pop_[cost_sort[0:self.n_elite]]
            orig[0:self.n_elite] = (cost_sort[0:self.n_elite] + i * self.n_chromosomes).astype(np.str)

            #Cumulative probability of selection as parent
            zcost = (self.cost_ - np.average(self.cost_)) / np.std(self.cost_)
            pzcost = 1 - norm.cdf(zcost)
            pcost = np.cumsum(pzcost / sum(pzcost))

            #Select parents & match
            numparents = self.n_chromosomes - self.n_elite
            #TODO: Add random state
            rand_parents = np.random.rand(numparents, 2)
            parents = np.zeros(rand_parents.shape, dtype=np.int)
            for parent1 in range(numparents):
                for parent2 in range(2):
                    parents[parent1, parent2] = np.searchsorted(pcost, rand_parents[parent1, parent2])

                if self.type_ == 'binary':
                    #Binary
                    #random single point matching
                    rand_match = int(np.random.rand() * self.n_parameters)
                    child = self.pop_[parents[parent1, 0]]
                    child[rand_match:] = self.pop_[parents[parent1, 1]]
                else:
                    #Continious
                    rand_match = np.random.rand(self.n_parameters)
                    child = self.pop_[parents[parent1, 0]] * rand_match
                    child += (1 - rand_match) * self.pop_[parents[parent1, 1]]

                new_pop[self.n_elite + parent1] = child

            orig[self.n_elite:] = [','.join(row.astype(np.str)) for row in (parents + i * self.n_chromosomes)]

            #Mutate
            m_rand = np.random.rand(self.n_chromosomes, self.n_parameters)
            m_rand[0:self.n_elite] = 1.0
            mutations = m_rand <= self.per_mutations
            num_mutations = np.count_nonzero(mutations)

            if self.type_ == 'binary':
                new_pop[mutations] = int(new_pop[mutations] == 0)
            else:
                new_pop[mutations] = self._random(num_mutations)[:, 0]

            rows_mutations = np.any(mutations, axis=1)
            orig[rows_mutations] = add(orig[rows_mutations], ['_M'] * np.count_nonzero(rows_mutations))

            # Replace replicates with random
            temp_unique = np.ascontiguousarray(new_pop).view(np.dtype((np.void,
                                                                       new_pop.dtype.itemsize * new_pop.shape[1])))
            _, temp_unique_idx = np.unique(temp_unique, return_index=True)
            n_replace = self.n_chromosomes - temp_unique_idx.shape[0]
            if n_replace > 0:
                temp_unique_replace = np.ones(self.n_chromosomes, dtype=np.bool)
                temp_unique_replace[:] = True
                temp_unique_replace[temp_unique_idx] = False
                new_pop[temp_unique_replace] = self._random(n_replace)
                orig[temp_unique_replace] = '-1'

            self.pop_ = new_pop
            self.cost_ = self.fitness_function(self.pop_, *self.fargs)

            filter_iter = range((i + 1) * self.n_chromosomes, (i + 2) * self.n_chromosomes)
            self.hist_.loc[filter_iter, 'iter'] = i + 1
            self.hist_.loc[filter_iter, 'cost'] = self.cost_
            self.hist_.loc[filter_iter, _cols_x] = self.pop_
            self.hist_.loc[filter_iter, 'orig'] = orig

        best = np.argmin(self.cost_)
        self.x = self.pop_[best]
        self.x_cost = self.cost_[best]
