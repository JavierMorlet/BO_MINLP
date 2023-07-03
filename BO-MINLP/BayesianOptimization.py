# Module for Bayesian Optimization

# Bayseian Optimization for mixed integer non linear programming (MINLP) problems
# Library: GPy
# Surrogate model: Gaussian Process (GP), and Student-t Process (TP)
# Adquisition functions: Expected Improvement (EI) with exploration parameter, Upper Confidence Bound (UCB) with exploration parameter, and Thompson Sampling (TS)
# Domain reduction: from best point
# Integer variables: Random sampling with Uniform distribution and Bernoulli distribution
# Continuous variables: Random sampling with Uniform distribution and Gaussian distribution

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import GPy
from scipy.stats import norm, truncnorm, invgamma

# Define class for Bayesian Optimization with GPy
class BO_GPy:
    
    def __init__(self, f, g, h, bounds_x, bounds_y, kernel, n, m, 
                 n_restarts = 5, c = 50, Δc = 25, q = 2, points = 500, n_iter = 50, n_inner_iter = 50,
                 surrogate = "SGP", adquisition = 'EI', exploration_parameter = 0.1, 
                 sample_x = "Uniform", sample_y = "Uniform", param_est_bernoulli = None, param_est_gaussian = None, 
                 initialization = False, x_0 = None, y_0 = None, Ω_0 = None, 
                 early_stop = True, DomainReduction = True, verbose = False):
        self.f = f
        self.g = g
        self.h = h
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.kernel = kernel
        self.n = n
        self.m = m
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.c = c
        self.Δc = Δc
        self.q = q
        self.points = points
        self.n_inner_iter = n_inner_iter
        self.surrogate = surrogate
        self.adquisition = adquisition
        self.exploration_parameter = exploration_parameter
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.param_est_bernoulli = param_est_bernoulli
        self.param_est_gaussian = param_est_gaussian
        self.x_0 = x_0
        self.y_0 = y_0
        self.Ω_0 = Ω_0
        self.initialization = initialization
        self.early_stop = early_stop
        self.DomainReduction = DomainReduction
        self.verbose = verbose
        self.model = None
        self.nu = None
        self.x = None
        self.y = None
        self.xy = None
        self.Ω = None
        self.x_max = None
        self.y_max = None
        self.Ω_max = None
        self.p_max = None
        self.f_max = None
        self.x_temp = None
        self.y_temp = None
        self.Ω_temp = None
        self.i = None
    def fit(self):
        # Penalty function p(x,y) = \sum_{i=1}^{r}(max(0, h_i(x,y))^q) + \sum_{i=1}^{s}(abs(g_i(x))^q)
        def p(x, y, q):
            G = np.abs(self.g(x, y))**q
            H = np.fmax(0, self.h(x, y))**q
            return np.sum(G, axis=1) + np.sum(H, axis=1)
        # Define the new objective function Omega
        def Omega(x, y, c, q):
            return self.f(x, y) - c*p(x, y, q)
    # Define surrogate model
        def surrogate_model(xy, Ω, kernel, surrogate):
            if surrogate == "GP":
                model = GPy.models.GPRegression(xy, Ω, kernel)
            elif surrogate == "STP":
                model = GPy.models.TPRegression(xy, Ω, kernel, deg_free=self.nu)
            elif surrogate == "SGP":
                model = GPy.models.SparseGPRegression(xy, Ω, kernel)
            else:
                print('Error: surrogate model not defined')
            return model
    # Define surrogate predictions
        def surrogate_predict(model, xy):
            return model.predict(xy)
    # Define surrogate training model
        def surrogate_train(model, n_restarts):
            # optimize the gpy model for the current dataset with lbfgsb
            model.optimize(optimizer='lbfgsb', max_iters=1000, messages=False)
            model.optimize_restarts(num_restarts=n_restarts, verbose=False)
            return model
    # Define acquisition function with EI
        # EI with GP
        def adquisition_ei(μ_best, xy_sample, model):
            μ_sample, σ_sample = surrogate_predict(model, xy_sample)
            with np.errstate(divide='warn'):
                imp = μ_sample - μ_best - self.exploration_parameter
                Z = imp / σ_sample
                ei = imp * norm.cdf(Z) + σ_sample * norm.pdf(Z)
                ei[σ_sample == 0.0] = 0.0
            return ei
        # EI with STP
        def adquisition_ei_STP(μ_best, xy_sample, model, nu):
            μ_sample, σ_sample = surrogate_predict(model, xy_sample)
            with np.errstate(divide='warn'):
                imp = μ_sample - μ_best - self.exploration_parameter
                Z = imp / σ_sample
                # (imp)Φt(z)+ ν/(ν−1)*(1 + z^2/nu)*σ*φt(z)
                ei = imp * norm.cdf(Z) + (nu/(nu-1))*(1 + Z**2/nu)*σ_sample*norm.pdf(Z)
                ei[σ_sample == 0.0] = 0.0
            return ei
    # Define acquisition function with UCB
        def adquisition_ucb(xy_sample, model):
            μ_sample, σ_sample = surrogate_predict(model, xy_sample)
            return μ_sample + self.exploration_parameter * σ_sample
    # Define acquisition function with TS
        def adquisition_ts(xy_sample, model, thompson_samples=5):
            # Reduce the number of points if the number of points is too high
            size = xy_sample.shape[0]
            if size > 1000:
                size = int(size // 100)
                xy_sample = xy_sample[np.random.choice(xy_sample.shape[0], size=size, replace=False), :]
            # Sample from the posterior distribution
            posterior_sample = model.posterior_samples_f(xy_sample, full_cov=True, size=thompson_samples)
            return np.mean(posterior_sample, axis=0)
    # Define random sampling with Gaussian distribution for continuous variables
        def random_sampling_gaussian_x(x_l, x_u, μ, σ):
            clil_x_l = (x_l - μ) / σ
            clil_x_u = (x_u - μ) / σ
            return truncnorm.rvs(clil_x_l, clil_x_u, loc=μ, scale=σ, size=(self.points, self.n))
    # Define parameter estimation of Bernoulli distribution
        def parameter_bernoulli(y, param_est_bernoulli, α=None, β=None):
            if param_est_bernoulli == "MLE":
                θ = np.mean(y, axis=0)
                return θ
            elif param_est_bernoulli == "Bayes":
                Σ = np.sum(y, axis=0)
                n = y.shape[0]
                α_n = α + Σ
                β_n = β + n - Σ
                θ = (α_n + Σ) / (α_n + β_n + n)
                return θ, α_n, β_n
    # Define parameter estimation of Gaussian distribution
        def parameter_gaussian(x, param_est_gaussian, α=None, β=None, κ=None, μ=None, σ=None):
            if param_est_gaussian == "MLE":
                μ = np.mean(x, axis=0)
                σ = np.std(x, axis=0)
                return μ, σ
            elif param_est_gaussian == "Bayes":
                Σ = np.sum(x, axis=0)
                n = x.shape[0]
                α_n = α + n/2
                κ_n = κ + n
                β_n = β + 1/2 * np.sum(x**2, axis=0) + (κ*n*(μ-Σ/n)**2) / (2*(κ+n))
                μ_n = (κ*μ + Σ) / (κ+n)
                σ_n = np.sqrt(β_n * (κ+n) / (α_n * κ_n))
                return μ_n, σ_n, α_n, β_n, κ_n
    # Define next sample with random sampling with Uniform distribution, Bernoulli distribution, and Gaussian distribution
        def next_sample(x_l, x_u, sample_x, sample_y, θ=None, μ=None, σ=None):
            if sample_y == 'Uniform':
                y_sample = np.random.randint(self.bounds_y[1]+1, size = (self.points, self.m))
            elif sample_y == 'Bernoulli':
                y_sample = np.random.binomial(self.bounds_y[1], θ, size = (self.points, self.m))
            else:
                print('Error: sample function not defined')
            if sample_x == 'Uniform':
                x_sample = np.random.Uniform(x_l, x_u, size=(self.points, self.n))
            elif sample_x == 'Gaussian':
                x_sample = random_sampling_gaussian_x(x_l, x_u, μ, σ)
            else:
                print('Error: sample function not defined')
            return x_sample, y_sample
    # Define best points for adquisition function
        def AF_best(model, x_sample, y_sample, adquisition, surrogate_model, μ_best = None, nu = None):
            xy_sample = np.hstack((x_sample, y_sample))
            if (adquisition == 'EI' and surrogate_model == 'GP') or (adquisition == 'EI' and surrogate_model == 'SGP'):
                score = adquisition_ei(μ_best, xy_sample, model)
            elif adquisition == 'EI' and surrogate_model == 'STP':
                score = adquisition_ei_STP(μ_best, xy_sample, model, nu)
            elif adquisition == 'UCB':
                score = adquisition_ucb(xy_sample, model)
            elif adquisition == 'TS':
                score = adquisition_ts(xy_sample, model)
            else:
                print('Error: adquisition function not defined')
            ix = np.argmax(score)
            return x_sample[ix, :], y_sample[ix, :]
    # Define initial sample for continuous variables if init is not defined
        def init_sample(max_iter=50):
            x = np.random.Uniform(self.bounds_x[0], self.bounds_x[1], size=(self.points, self.n))
            y = np.random.randint(self.bounds_y[1]+1, size = (self.points, self.m))
            Ω = Omega(x, y, self.c, self.q)
            ix = np.argmax(Ω)
            x_max, y_max, Ω_max = x[ix, :], y[ix, :], Ω[ix]
            for i in range(max_iter):
                x = np.random.Uniform(self.bounds_x[0], self.bounds_x[1], size=(self.points, self.n))
                y = np.random.randint(self.bounds_y[1]+1, size = (self.points, self.m))
                Ω = Omega(x, y, self.c, self.q)
                ix = np.argmax(Ω)
                x_new, y_new = x[ix, :], y[ix, :]
                Ω_new = np.max(Ω)
                x_max, y_max, Ω_max = np.vstack((x_max, x_new)), np.vstack((y_max, y_new)), np.vstack((Ω_max, Ω_new))
            return x_max, y_max, Ω_max
    # Define domain reduction from best point
        def domain_reduction(init_x_l, init_x_u, x_best, y_best, eps=0.75):
            if x_best is None:
                x_best = self.x[np.argmax(self.Ω), :]
            if y_best is None:
                y_best = self.y[np.argmax(self.Ω), :]
            if init_x_l is None:
                init_x_l = self.bounds_x[0]
            if init_x_u is None:
                init_x_u = self.bounds_x[1]
            x_r = init_x_u - init_x_l
            x_l = x_best - eps*(x_r)
            x_u = x_best + eps*(x_r)
            x_l_new = np.empty(self.n)
            x_u_new = np.empty(self.n)
        # Check bounds for x_l and x_u where x_l and x_u are numpy arrays
            for j in range(len(x_l)):
                if x_l[j] < self.bounds_x[0][j]:
                    x_l_new[j] = self.bounds_x[0][j]
                else:
                   x_l_new[j] = x_l[j]
                if x_u[j] > self.bounds_x[1][j]:
                    x_u_new[j] = self.bounds_x[1][j]
                else:
                    x_u_new[j] = x_u[j]
            return x_l_new, x_u_new
    # Define plot results for f, Ω and p
        def plot_results(Ω, p, f, savefig=False):
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].plot(Ω)
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel('Ω')
            ax[1].plot(p)
            ax[1].set_xlabel('Iterations')
            ax[1].set_ylabel('p')
            ax[1].set_ylim(0,1)
            ax[2].plot(f)
            ax[2].set_xlabel('Iterations')
            ax[2].set_ylabel('f')
            if savefig:
                plt.savefig('results.png')
            plt.show()
    # Main program
        # Define initial sample if initialisation is true
        if self.initialization:
            self.x, self.y, self.Ω = init_sample()
        else:
            if self.x_0 is None:
                self.x_0 = np.random.uniform(self.bounds_x[0], self.bounds_x[1], size=(50, n))
            elif self.y_0 is None:
                self.y_0 = np.random.randint(self.bounds_y[1]+1, size = (50, m))
            elif self.Ω_0 is None:
                self.Ω_0 = Omega(self.x_0, self.y_0, self.c, self.q)
            self.x, self.y, self.Ω = self.x_0, self.y_0, self.Ω_0
        self.Ω = self.Ω.reshape(-1,1)
        self.xy = np.hstack((self.x, self.y))
        # Define surrogate model
        if self.surrogate is None:
            self.surrogate == "SGP"
        if self.surrogate == "STP":
            self.nu = 5
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=n_tot, variance=1.)
        self.model = surrogate_model(self.xy, self.Ω, self.kernel, self.surrogate)
        # Train surrogate model
        self.model = surrogate_train(self.model, self.n_restarts)
        # Initialise if sample is not defined
        if self.sample_y is None:
            self.sample_y = 'Bernoulli'
        if self.sample_x is None:
            self.sample_x = "Gaussian"
        if self.sample_y == 'Bernoulli':
            if self.param_est_bernoulli is None:
                self.param_est_bernoulli = "Bayes"
        if self.sample_x == "Gaussian":
            if self.param_est_gaussian is None:
                self.param_est_gaussian = "Bayes"
        # Define initial values for parameters of Bernoulli and Gaussian distributions
        if self.sample_y == 'Bernoulli':
            if self.param_est_bernoulli == "Bayes":
                α_b0 = 0
                β_b0 = 0
        if self.sample_x == "Gaussian":
            if self.param_est_gaussian == "Bayes":
                α_g0 = 0
                β_g0 = 0
                κ_g0 = self.x.shape[1] + 1
                μ_g0 = np.mean(self.x)
                σ_g0 = np.std(self.x)
        # Initialize if acquisition function is not defined
        if self.adquisition is None:
            self.adquisition = 'EI'
        #Set initial values for x_l and x_u, and c_0
        x_l, x_u = self.bounds_x[0], self.bounds_x[1]       
        self.i = 0
    # Main loop
        while self.i < self.n_iter:
            # Update c value
            self.c += self.Δc
            # Store best point
            ix = np.argmax(self.Ω)
            # Save best point
            if self.i == 0:
                self.x_max, self.y_max, self.Ω_max = self.x[ix, :], self.y[ix, :], self.Ω[ix, :]
                x_best, y_best, Ω_best = self.x_max, self.y_max, self.Ω_max
                self.p_max = p(x_best.reshape(1,-1), y_best.reshape(1,-1), self.q)
                self.f_max = self.f(x_best.reshape(1,-1), y_best.reshape(1,-1))
            else:
                self.x_max, self.y_max, self.Ω_max = np.vstack((self.x_max, self.x[ix, :])), np.vstack((self.y_max, self.y[ix, :])), np.vstack((self.Ω_max, self.Ω[ix, :]))
                x_best, y_best, Ω_best = self.x_max[-1,:], self.y_max[-1,:], self.Ω_max[-1,:]
                self.p_max = np.vstack((self.p_max, p(x_best.reshape(1,-1), y_best.reshape(1,-1), self.q)))
                self.f_max = np.vstack((self.f_max, self.f(x_best.reshape(1,-1), y_best.reshape(1,-1))))
            # Initialize μ_best if Adquisition function is EI
            if self.adquisition == 'EI':
                μ, _ = surrogate_predict(self.model, self.xy)
                μ_best = np.max(μ)
            # Update parameters of Bernoulli and Gaussian distributions
            if self.sample_y == 'Bernoulli':
                if self.param_est_bernoulli == "Bayes":
                    θ, α_b, β_b = parameter_bernoulli(self.y, self.param_est_bernoulli, α_b0, β_b0)
                    α_b0, β_b0 = α_b, β_b
                elif self.param_est_bernoulli == "MLE":
                    θ = parameter_bernoulli(self.y, self.param_est_bernoulli)
                else:
                    print("Error: param_est_bernoulli must be 'MLE' or 'Bayes'")
            if self.sample_x == 'Gaussian':
                if self.param_est_gaussian == "Bayes":
                    μ, σ, α_g, β_g, κ_g = parameter_gaussian(self.x, self.param_est_gaussian, α_g0, β_g0, κ_g0, μ_g0, σ_g0)
                    μ_g0, σ_g0, α_g0, β_g0 = μ, σ, α_g, β_g
                    κ_g0 = self.x.shape[1] + 1
                elif self.param_est_gaussian == "MLE":
                    μ, σ = parameter_gaussian(self.x, self.param_est_gaussian)
                else:
                    print("Error: param_est_gaussian must be 'MLE' or 'Bayes'")
            # Reduce domain of the continuous variables if Domain Reduction is True
            if self.DomainReduction:
                x_l_new, x_u_new = domain_reduction(init_x_l=x_l, init_x_u=x_u, x_best=x_best, y_best=y_best)
            else:
                x_l_new, x_u_new = x_l, x_u
            self.x_temp, self.y_temp, self.Ω_temp = np.empty((0, self.n)), np.empty((0, self.m)), np.empty((0, 1))
            # Print iteration results if verbose
            if self.verbose:
                print('****** iteration *****', self.i)
                print('Ω_max: % 0.2f' % Ω_best, 'x_max', np.around(x_best, decimals=2), 'y_max', y_best)  
        # Inner loop          
            for j in range(self.n_inner_iter):
                # Sample new point
                if self.sample_y == 'Uniform' and self.sample_x == 'Uniform': 
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y)
                elif self.sample_y == 'Bernoulli' and self.sample_x == 'Uniform':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, θ = θ)
                elif self.sample_y == 'Uniform' and self.sample_x == 'Gaussian':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, μ = μ, σ = σ)
                elif self.sample_y == 'Bernoulli' and self.sample_x == 'Gaussian':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, θ = θ, μ = μ, σ = σ)
                # Compute adquisition function
                if self.adquisition == 'EI':
                    if self.surrogate == 'STP':
                        x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate, μ_best=μ_best, nu = self.nu)
                    else:
                        x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate, μ_best=μ_best)
                else:
                    x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate)
                # Compute objective function
                Ω_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), c = self.c, q = self.q)
                # Break if new point is better than the best point
                if Ω_new > Ω_best:
                    self.i += 1
                    break
                # Save temporary results
                else:
                    self.x_temp, self.y_temp, self.Ω_temp = np.vstack((self.x_temp, x_new)), np.vstack((self.y_temp, y_new)), np.vstack((self.Ω_temp, Ω_new))
                # Return best temporary point if the inner loop is finished
                if j == (self.n_inner_iter - 1):
                    self.i += 1
                    ix_temp = np.argmax(self.Ω_temp)
                    x_new, y_new = self.x_temp[ix_temp, :], self.y_temp[ix_temp, :]
                    Ω_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), c = self.c, q = self.q)
            # Update training set
            self.x, self.y = np.vstack((self.x, x_new)), np.vstack((self.y, y_new))
            self.xy = np.hstack((self.x, self.y))
            self.Ω = Omega(self.x, self.y, c = self.c, q = self.q)
            self.Ω = self.Ω.reshape(-1,1)
            # Update bounds of the continuous variables
            x_l, x_u = x_l_new, x_u_new
            self.model = surrogate_model(self.xy, self.Ω, self.kernel, self.surrogate)
            self.model = surrogate_train(self.model, self.n_restarts)
            # Check if iterations are not improving
            if self.early_stop:
                if len(self.p_max) > 5:
                    # Break if p_max is less than tolerance and f_max is not improving for 10 iterations
                    if self.p_max[-1] < 1e-3 and np.abs(np.sum(np.diff(self.f_max[-10:], axis=0))) < 1e-6:
                        break
        # Print final results if verbose
        if self.verbose:
            print('****** Final results *****')
            print('Ω_max: % 0.2f' % Ω_best, 'x_max', np.around(x_best, decimals=2), 'y_max', y_best)
            plot_results(self.Ω_max, self.p_max, self.f_max)
    # Return best point
        return self.Ω_max, self.p_max, self.f_max, self.x_max, self.y_max, self.x, self.y, self.Ω