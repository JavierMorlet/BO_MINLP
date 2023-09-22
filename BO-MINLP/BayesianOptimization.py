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
    
    def __init__(self, f, g, h, noise_f, noise_g, noise_h,
                 bounds_x, bounds_y, kernel, n, m,
                 n_restarts = 5, lamb = 100, delta_lamb = 5, q = 2, points = 1000, t_max = 100, j_max = 50,
                 surrogate = "SGP", adquisition = 'EI', xi_AF = 0.1, 
                 sample_x = "Gaussian", sample_y = "Bernoulli", param_est_bernoulli = "Bayes", param_est_gaussian = "Bayes",
                 initialization = False, D_0 = None, early_stop = True, DomainReduction = False, verbose = False):
        
        self.f = f
        self.g = g
        self.h = h
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.kernel = kernel
        self.n = n
        self.m = m
        self.noise_f = noise_f
        self.noise_g = noise_g
        self.noise_h = noise_h
        self.n_restarts = n_restarts
        self.t_max = t_max
        self.lamb = lamb
        self.delta_lamb = delta_lamb
        self.q = q
        self.points = points
        self.j_max = j_max
        self.surrogate = surrogate
        self.adquisition = adquisition
        self.xi_AF = xi_AF
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.param_est_bernoulli = param_est_bernoulli
        self.param_est_gaussian = param_est_gaussian
        self.initialization = initialization
        self.early_stop = early_stop
        self.DomainReduction = DomainReduction
        self.verbose = verbose
        self.model = None
        self.nu = None
        self.x = None
        self.y = None
        self.xy = None
        self.omega = None
        self.D_0 = D_0
        self.x_0 = None
        self.y_0 = None
        self.omega_0 = None
        self.x_best = None
        self.y_best = None
        self.omega_best = None
        self.D_best = None
        self.p_best = None
        self.f_best = None
        self.x_temp = None
        self.y_temp = None
        self.omega_temp = None
        self.D_temp = None
        self.t = None

    def fit(self):

        # Penalty function p(x,y) = \sum_{i=1}^{r}(max(0, h_i(x,y))^q) + \sum_{i=1}^{s}(abs(g_i(x))^q)
        def p(x, y, q, noise_g=None, noise_h=None):
            if noise_g != None:
                G = np.abs(self.g(x, y, noise_g))**q
            elif noise_g == None:
                G = np.abs(self.g(x, y))**q
            if noise_h != None:
                H = np.fmax(0, self.h(x, y, noise_h))**q
            elif noise_h == None:
                H = np.fmax(0, self.h(x, y))**q

            return np.sum(G, axis=1) + np.sum(H, axis=1)
        
        # Define the new objective function Omega
        def Omega(x, y, lamb, q, noise_f=None, noise_g=None, noise_h=None):
            if noise_f is not None:
                F = self.f(x, y, noise_f)
            else:
                F = self.f(x, y)
            if noise_g is not None and noise_h is not None:
                P = p(x, y, q, noise_g, noise_h)
            elif noise_g is not None:
                P = p(x, y, q, noise_g, None)
            elif noise_h is not None:
                P = p(x, y, q, None, noise_h)
            else:
                P = p(x, y, q)

            return F - lamb*P
        
        # Define surrogate model
        def surrogate_model(xy, omega, kernel, surrogate):
            if surrogate == "GP":
                model = GPy.models.GPRegression(xy, omega, kernel)
            elif surrogate == "STP":
                model = GPy.models.TPRegression(xy, omega, kernel, deg_free=self.nu)
            elif surrogate == "SGP":
                model = GPy.models.SparseGPRegression(xy, omega, kernel)
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
    
        # Define initial sample for continuous variables if init is not defined
        def init_sample(max_iter=50, noise_f=None, noise_g=None, noise_h=None):
            D_0 = []
            x_samp = np.random.uniform(self.x_l, self.x_u, size=(self.points, self.n))
            y_samp = np.random.randint(self.y_l, self.y_u+1, size = (self.points, self.m))
            if noise_f is not None and noise_g is not None and noise_h is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_g = noise_g, noise_h = noise_h)
            elif noise_f is not None and noise_g is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_g = noise_g)
            elif noise_f is not None and noise_h is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_h = noise_h)
            elif noise_g is not None and noise_h is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_g = noise_g, noise_h = noise_h)
            elif noise_f is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f)
            elif noise_g is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_g = noise_g)
            elif noise_h is not None:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_h = noise_h)
            else:
                omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q)
            ix = np.argmax(omega_samp)
            x_max, y_max, omega_max = x_samp[ix, :], y_samp[ix, :], omega_samp[ix]
            for i in range(max_iter):
                x_samp = np.random.uniform(self.x_l, self.x_u, size=(self.points, self.n))
                y_samp = np.random.randint(self.y_l, self.y_u+1, size = (self.points, self.m))
                if noise_f is not None and noise_g is not None and noise_h is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_g = noise_g, noise_h = noise_h)
                elif noise_f is not None and noise_g is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_g = noise_g)
                elif noise_f is not None and noise_h is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f, noise_h = noise_h)
                elif noise_g is not None and noise_h is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_g = noise_g, noise_h = noise_h)
                elif noise_f is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_f = noise_f)
                elif noise_g is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_g = noise_g)
                elif noise_h is not None:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q, noise_h = noise_h)
                else:
                    omega_samp = Omega(x_samp, y_samp, lamb = self.lamb, q = self.q)
                ix = np.argmax(omega_samp)
                # Add new point to D_0
                D_0 = np.vstack((D_0, np.hstack((x_samp[ix, :], y_samp[ix, :], omega_samp[ix]))))

            return np.array(D_0)
        
        # Define random sampling with Gaussian distribution for continuous variables
        def random_sampling_gaussian_x(x_l, x_u, mu, sigma):
            clil_x_l = (x_l - mu) / sigma
            clil_x_u = (x_u - mu) / sigma

            return truncnorm.rvs(clil_x_l, clil_x_u, loc=mu, scale=sigma, size=(self.points, self.n))
        
        # Define parameter estimation of Bernoulli distribution
        def parameter_bernoulli(y, param_est_bernoulli, alpha=None, beta=None):
            if param_est_bernoulli == "MLE":
                theta = np.mean(y, axis=0)

                return theta
            
            elif param_est_bernoulli == "Bayes":
                sigma = np.sum(y, axis=0)
                n = y.shape[0]
                alpha_n = alpha + sigma
                beta_n = beta + n - sigma
                theta = (alpha_n + sigma) / (alpha_n + beta_n + n)

                return theta, alpha_n, beta_n
        
        # Define parameter estimation of Gaussian distribution
        def parameter_gaussian(x, param_est_gaussian, alpha=None, beta=None, kappa=None, mu=None, sigma=None):
            if param_est_gaussian == "MLE":
                mu = np.mean(x, axis=0)
                sigma = np.std(x, axis=0)

                return mu, sigma
            
            elif param_est_gaussian == "Bayes":
                sigma = np.sum(x, axis=0)
                n = x.shape[0]
                alpha_n = alpha + n/2
                kappa_n = kappa + n
                beta_n = beta + 1/2 * np.sum(x**2, axis=0) + (kappa*n*(mu-sigma/n)**2) / (2*(kappa+n))
                mu_n = (kappa*mu + sigma) / (kappa+n)
                sigma_n = np.sqrt(beta_n * (kappa+n) / (alpha_n * kappa_n))

                return mu_n, sigma_n, alpha_n, beta_n, kappa_n
        
        # Define next sample with random sampling with Uniform distribution, Bernoulli distribution, and Gaussian distribution
        def next_sample(x_l, x_u, sample_x, sample_y, theta=None, mu=None, sigma=None):
            if sample_y == 'Uniform':
                y_sample = np.random.randint(self.y_l, self.y_u+1, size = (self.points, self.m))
            elif sample_y == 'Bernoulli':
                y_sample = np.random.binomial(self.y_u, theta, size = (self.points, self.m))
            else:
                print('Error: sample function not defined')
            if sample_x == 'Uniform':
                x_sample = np.random.uniform(x_l, x_u, size=(self.points, self.n))
            elif sample_x == 'Gaussian':
                x_sample = random_sampling_gaussian_x(x_l, x_u, mu, sigma)
            else:
                print('Error: sample function not defined')

            return x_sample, y_sample
        
        # Define acquisition function 
        
        # EI with GP
        def adquisition_ei(mu_best, xy_sample, model, xi):
            mu_sample, sigma_sample = surrogate_predict(model, xy_sample)
            with np.errstate(divide='warn'):
                imp = mu_sample - mu_best - xi
                Z = imp / sigma_sample
                ei = imp * norm.cdf(Z) + sigma_sample * norm.pdf(Z)
                ei[sigma_sample == 0.0] = 0.0

            return ei
        
        # EI with STP
        def adquisition_ei_STP(mu_best, xy_sample, model, nu, xi):
            mu_sample, sigma_sample = surrogate_predict(model, xy_sample)
            with np.errstate(divide='warn'):
                imp = mu_sample - mu_best - xi
                Z = imp / sigma_sample
                ei = imp * norm.cdf(Z) + (nu/(nu-1))*(1 + Z**2/nu)*sigma_sample*norm.pdf(Z)
                ei[sigma_sample == 0.0] = 0.0

            return ei
        
        # UCB
        def adquisition_ucb(xy_sample, model, xi):
            mu_sample, sigma_sample = surrogate_predict(model, xy_sample)
            
            return mu_sample + xi * sigma_sample
        
        # TS
        def adquisition_ts(xy_sample, model, thompson_samples=5):
            # Reduce the number of points if the number of points is too high
            size = xy_sample.shape[0]
            if size > 1000:
                size = int(size // 100)
                xy_sample = xy_sample[np.random.choice(xy_sample.shape[0], size=size, replace=False), :]
            # Sample from the posterior distribution
            posterior_sample = model.posterior_samples_f(xy_sample, full_cov=True, size=thompson_samples)

            return np.mean(posterior_sample, axis=0)
        
        # Define best points for adquisition function
        def AF_best(model, x_sample, y_sample, adquisition, surrogate_model, mu_best = None, nu = None, xi_AF = None):
            xy_sample = np.hstack((x_sample, y_sample))
            if (adquisition == 'EI' and surrogate_model == 'GP') or (adquisition == 'EI' and surrogate_model == 'SGP'):
                score = adquisition_ei(mu_best, xy_sample, model, xi_AF)
            elif adquisition == 'EI' and surrogate_model == 'STP':
                score = adquisition_ei_STP(mu_best, xy_sample, model, nu, xi_AF)
            elif adquisition == 'UCB':
                score = adquisition_ucb(xy_sample, model, xi_AF)
            elif adquisition == 'TS':
                score = adquisition_ts(xy_sample, model)
            else:
                print('Error: adquisition function not defined')
            ix = np.argmax(score)

            return x_sample[ix, :], y_sample[ix, :]
        
        # Define domain reduction from best point
        def domain_reduction(x_l_0, x_u_0, x_best, y_best, eta=0.9):
            if x_best is None:
                x_best = self.x[np.argmax(self.omega), :]
            if y_best is None:
                y_best = self.y[np.argmax(self.omega), :]
            if x_l_0 is None:
                x_l_0 = self.x_l
            if x_u_0 is None:
                x_u_0 = self.x_u
            x_r = x_u_0 - x_l_0
            x_l = x_best - eta*(x_r)
            x_u = x_best + eta*(x_r)
            x_l_new = np.empty(self.n)
            x_u_new = np.empty(self.n)
            # Check bounds for x_l and x_u
            for j in range(self.n_c):
                if x_l[j] < self.x_l[j]:
                    x_l_new[j] = self.x_l[j]
                else:
                   x_l_new[j] = x_l[j]
                if x_u[j] > self.x_u[j]:
                    x_u_new[j] = self.x_u[j]
                else:
                    x_u_new[j] = x_u[j]

            return x_l_new, x_u_new
        
        # Define plot results for f, omega and p
        def plot_results(omega, p, f, savefig=False):
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].plot(omega)
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel('omega')
            ax[1].plot(p)
            ax[1].set_xlabel('Iterations')
            ax[1].set_ylabel('p')
            ax[1].set_ylim(0,0.5)
            ax[2].plot(f)
            ax[2].set_xlabel('Iterations')
            ax[2].set_ylabel('f')
            if savefig:
                plt.savefig('results.png')
            plt.show()
    
    # *************** Main program ********************

        # Define number of variables
        self.n_vars = self.n + self.m
        # Define values for bounds 
        self.x_l, self.x_u = self.bounds_x[0], self.bounds_x[1]
        self.y_l, self.y_u = self.bounds_y[0], self.bounds_y[1]
        # Define initial sample if initialization is true
        if self.initialization:
            self.D_0 = init_sample()
        # Define initial values for x, y, and omega from D_0
        self.N_D = self.D_0.shape[0]
        self.x, self.y, self.omega = self.D_0[:,0:self.n], self.D_0[:,self.n:self.n_vars], self.D_0[:,-1]
        self.omega = self.omega.reshape(-1,1)
        self.xy = np.hstack((self.x, self.y))
        # Define surrogate model
        if self.surrogate == "STP":
            self.nu = 5
        # Define kernel if kernel is "RBF"
        if self.kernel == "RBF":
            self.kernel = GPy.kern.RBF(self.n_vars, lengthscale=1)
        # Define initial values for parameters of Bernoulli and Gaussian distributions if param_est is "Bayes"
        if self.sample_y == 'Bernoulli' and self.param_est_bernoulli == "Bayes":
            alpha_b0 = 0
            beta_b0 = 0
        if self.sample_x == "Gaussian" and self.param_est_gaussian == "Bayes":
            alpha_g0 = 0
            beta_g0 = 0
            kappa_g0 = self.N_D + 1
            mu_g0 = np.mean(self.x)
            sigma_g0 = np.std(self.x)
        #Set initial values for x_l and x_u
        x_l_t, x_u_t = self.x_l, self.x_u
        # Initialize parameters for loop
        self.t = 0
        self.D = self.D_0
        # Initialize lists for results
        self.omega_best = np.max(self.omega)
        ix = np.argmax(self.omega)
        self.x_best, self.y_best = self.x[ix, :], self.y[ix, :]
        if self.noise_f != None:
            noise_f = self.noise_f()
        else:
            noise_f = None
        if self.noise_g != None:
            noise_g = self.noise_g()
        else:
            noise_g = None
        if self.noise_h != None:
            noise_h = self.noise_h()
        else:
            noise_h = None
        if self.noise_g != None and self.noise_h != None:
            self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, noise_g, noise_h)
        elif self.noise_g != None:
            self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, noise_g, None)
        elif self.noise_h != None:
            self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, None, noise_h)
        else:
            self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, None, None)
        if self.noise_f != None:
            self.f_best = self.f(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), noise_f)
        else:
            self.f_best = self.f(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1))
        self.D_best = np.hstack((self.x_best, self.y_best, self.omega_best, self.p_best, self.f_best))

    # Main loop
        while self.t < self.t_max:
            # Update model with new data
            self.model = surrogate_model(self.xy, self.omega, self.kernel, self.surrogate)
            # Train surrogate model
            self.model = surrogate_train(self.model, self.n_restarts)
            # Update parameters of Bernoulli and Gaussian distributions
            if self.sample_y == 'Bernoulli':
                if self.param_est_bernoulli == "Bayes":
                    theta, alpha_b, beta_b = parameter_bernoulli(self.y, self.param_est_bernoulli, alpha_b0, beta_b0)
                    alpha_b0, beta_b0 = alpha_b, beta_b
                elif self.param_est_bernoulli == "MLE":
                    theta = parameter_bernoulli(self.y, self.param_est_bernoulli)
                else:
                    print("Error: param_est_bernoulli must be 'MLE' or 'Bayes'")
            if self.sample_x == 'Gaussian':
                if self.param_est_gaussian == "Bayes":
                    mu, sigma, alpha_g, beta_g, kappa_g = parameter_gaussian(self.x, self.param_est_gaussian, alpha_g0, beta_g0, kappa_g0, mu_g0, sigma_g0)
                    mu_g0, sigma_g0, alpha_g0, beta_g0 = mu, sigma, alpha_g, beta_g
                    kappa_g0 = self.x.shape[1] + 1
                elif self.param_est_gaussian == "MLE":
                    mu, sigma = parameter_gaussian(self.x, self.param_est_gaussian)
                else:
                    print("Error: param_est_gaussian must be 'MLE' or 'Bayes'")
            # Reduce domain of the continuous variables if Domain Reduction is True
            if self.DomainReduction:
                x_l_new, x_u_new = domain_reduction(init_x_l = x_l_t, init_x_u = x_u_t, x_best = self.x_best, y_best = self.y_best)
            else:
                x_l_new, x_u_new = self.x_l, self.x_u
            # Initialize mu_best if Adquisition function is EI
            if self.adquisition == 'EI':
                mu_EI, _ = surrogate_predict(self.model, self.xy)
                mu_best = np.max(mu_EI)
            # Update penalty parameter 
            self.lamb += self.delta_lamb
            # Update noise
            if self.noise_f is not None:
                noise_f = self.noise_f()
            if self.noise_g is not None:
                noise_g = self.noise_g()
            if self.noise_h is not None:
                noise_h = self.noise_h()
            # Initialize parameters for Inner loop
            self.D_temp = np.empty((0, self.n_vars+1))
            # Print results if verbose
            if self.verbose:
                print('****** iteration *****', self.t)
                print('omega_max: % 0.2f' % self.omega_best, 'x_max', np.around(self.x_best, decimals=2), 'y_max', self.y_best)  
            # *** Inner loop *** 
            for j in range(self.j_max):
                # Sample new point from distribution
                if self.sample_y == 'Uniform' and self.sample_x == 'Uniform': 
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y)
                elif self.sample_y == 'Bernoulli' and self.sample_x == 'Uniform':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, theta = theta)
                elif self.sample_y == 'Uniform' and self.sample_x == 'Gaussian':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, mu = mu, sigma = sigma)
                elif self.sample_y == 'Bernoulli' and self.sample_x == 'Gaussian':
                    x_sample, y_sample = next_sample(x_l_new, x_u_new, self.sample_x, self.sample_y, theta = theta, mu = mu, sigma = sigma)
                # Compute adquisition function
                if self.adquisition == 'EI':
                    if self.surrogate == 'STP':
                        x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate, mu_best=mu_best, nu = self.nu, xi_AF = self.xi_AF)
                    else:
                        x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate, mu_best=mu_best, xi_AF = self.xi_AF)
                elif self.adquisition == 'UCB':
                    x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate, xi_AF = self.xi_AF)
                elif self.adquisition == 'TS':
                    x_new, y_new = AF_best(self.model, x_sample, y_sample, adquisition=self.adquisition, surrogate_model=self.surrogate)
                else:
                    exit('Error: adquisition function not defined')
                # Compute objective function
                if noise_f is not None and noise_g is not None and noise_h is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, noise_g, noise_h)
                elif noise_f is not None and noise_g is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, noise_g, None)
                elif noise_f is not None and noise_h is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, None, noise_h)
                elif noise_f is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, None, None)
                elif noise_g is not None and noise_h is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, noise_g, noise_h)
                elif noise_g is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, noise_g, None)
                elif noise_h is not None:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, None, noise_h)
                else:
                    omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q)
                # Break if new point is better than the best point
                if omega_new > self.omega_best:
                    self.t += 1
                    break
                # Else: Save temporary results in D_temp
                else:
                    self.D_temp = np.vstack((self.D_temp, np.hstack((x_new, y_new, omega_new))))
                # Return best temporary point if the inner loop is finished
                if j == (self.j_max - 1):
                    ix_temp = np.argmax(self.D_temp[:,-1])
                    x_new, y_new = self.D_temp[ix_temp, 0:self.n], self.D_temp[ix_temp, self.n:self.n_vars]
                    if noise_f is not None and noise_g is not None and noise_h is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, noise_g, noise_h)
                    elif noise_f is not None and noise_g is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, noise_g, None)
                    elif noise_f is not None and noise_h is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, None, noise_h)
                    elif noise_f is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, noise_f, None, None)
                    elif noise_g is not None and noise_h is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, noise_g, noise_h)
                    elif noise_g is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, noise_g, None)
                    elif noise_h is not None:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q, None, None, noise_h)
                    else:
                        omega_new = Omega(x_new.reshape(1,-1), y_new.reshape(1,-1), self.lamb, self.q)
                    self.t += 1
            # Update training set
            self.x, self.y = np.vstack((self.x, x_new)), np.vstack((self.y, y_new))
            self.xy = np.vstack((self.xy, np.hstack((x_new, y_new))))
            if noise_f is not None and noise_g is not None and noise_h is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, noise_f, noise_g, noise_h)
            elif noise_f is not None and noise_g is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, noise_f, noise_g, None)
            elif noise_f is not None and noise_h is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, noise_f, None, noise_h)
            elif noise_f is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, noise_f, None, None)
            elif noise_g is not None and noise_h is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, None, noise_g, noise_h)
            elif noise_g is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, None, noise_g, None)
            elif noise_h is not None:
                self.omega = Omega(self.x, self.y, self.lamb, self.q, None, None, noise_h)
            else:
                self.omega = Omega(self.x, self.y, self.lamb, self.q)
            self.omega = self.omega.reshape(-1,1)
            self.D = np.vstack((self.D, np.hstack((x_new, y_new, omega_new))))
            # Update bounds of the continuous variables
            x_l, x_u = x_l_new, x_u_new
            # Update best point
            self.omega_best = np.max(self.omega)
            ix = np.argmax(self.omega)
            self.x_best, self.y_best = self.x[ix, :], self.y[ix, :]
            if self.noise_g != None and self.noise_h != None:
                self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, noise_g, noise_h)
            elif self.noise_g != None:
                self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, noise_g, None)
            elif self.noise_h != None:
                self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q, None, noise_h)
            else:
                self.p_best = p(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), self.q)
            if self.noise_f != None:
                self.f_best = self.f(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1), noise_f)
            else:
                self.f_best = self.f(self.x_best.reshape(1,-1), self.y_best.reshape(1,-1))
            self.D_best = np.vstack((self.D_best, np.hstack((self.x_best, self.y_best, self.omega_best, self.p_best, self.f_best))))
            # *** Stop criteria *** 
            # Check if iterations are not improving
            if self.early_stop:
                if len(self.D_best[:,-2]) > 10:
                    # Break if p_max is less than tolerance and f_max is not improving for 10 iterations
                    if self.D_best[:,-2] < 1e-3 and np.abs(np.sum(np.diff(self.D_best[:,-1], axis=0))) < 1e-5:
                        break
        # Print final results if verbose
        if self.verbose:
            print('****** Final results *****')
            print('omega_max: % 0.2f' % self.omega_best, 'x_max', np.around(self.x_best, decimals=2), 'y_max', self.y_best)
            plot_results(self.D_best[:,-3], self.D_best[:,-2], self.D_best[:,-1])
    
    # Return D and D_best
        return self.D, self.D_best