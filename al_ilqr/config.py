from .utils import col


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def update(self, **kwarg):
        # Printing
        my_keys = [
            key
            for key in kwarg.keys()
            if key in self.keys() and self[key] != kwarg[key]
        ]

        if len(my_keys) > 0:
            print(
                f"\n{col.UNDERLINE}Updating: {self.get('_name', 'X')}-config {col.ENDC}\n"
            )
            print(
                f'{"key": ^15}: {col.FAIL} {"old": ^15} -> {col.OKGREEN} {"new":^15} {col.ENDC}'
            )
            print(
                f'{"-" * 15: ^15}  {col.FAIL} {"-" * 15: ^15}    {col.OKGREEN} {"-" * 15:^15} {col.ENDC}'
            )
            for key in my_keys:
                print(
                    f"{key: ^15}: {col.FAIL} {str(self[key]): ^15} -> {col.OKGREEN} {str(kwarg[key]):^15} {col.ENDC}"
                )

        # The important part
        for key in my_keys:
            # assert key in self.keys(), f"{key} is not a valid key."
            self[key] = kwarg[key]


class Config(DotDict):
    def __init__(self) -> None:
        super().__init__()
        # iLQR hyperparams
        self.T = 60  # timesteps

        #   convergence
        self.eps_cost = None  # cost tolerance [1e-2, 1e-4, 1e-8]       , HIGH
        self.eps_grad = None  # gradient tolerance 1e-5                 , MEDIUM
        self.iter_max = None  # max nr of backw/forward iter [50, 500]  , MEDIUM
        self.max_cost = 1e8  # max cost during rollout                  , MEDIUM

        #   line search
        self.iter_line_search = None  # Max nr of backtr line search [5,10, 20]  , LOW
        self.beta_1 = None  # lower bound for linesearch               , LOW
        self.beta_2 = None  # upper bound for linesearch               , LOW

        #   regularization
        self.reg_max = None  # Any further increase will saturate 1e8   , LOW
        self.reg_min = None  # Any reg below will round to 0   1e-8     , LOW
        self.reg_scale_factor = None  # how much reg is up/descaled (1,1.6,10)   , LOW
        self.reg_init = None  # Init value for reg Quu in backw pass 0   , LOW

        # Augmented Lagrangian hyperparameters
        #   convergence
        self.al_eps_cost = None  # difference in cost between iterations for convergence [1e2, 1e-4, 1e-8],         HIGH
        self.al_eps_uncon = None  # cost tolerance for the iLQR to trigger al update(can require less optimality),   HIGH

        self.al_cmax = None  # convergence when max constraint violation < cmax      [1e-2, 1e-4, 1e-8],        HIGH
        self.al_mu_max = None  # maximal penalty(high better conv), (low avoid ill)     1e8,                      LOW
        self.al_max_iter = None  # max nr of outer loop updates                          [10, 30, 100],             MEDIUM

        #   penalty
        self.al_psi = None  # penalty scaling                                       (1, 10, 100],              MEDIUM
        self.al_mu_init = None  # initial penalty                                       [1e-4, 1, 100],            VERY HIGH


class iLQRConfigDefault(DotDict):
    def __init__(self) -> None:
        super().__init__()
        self._name = "iLQR"

        # iLQR hyperparams
        self.T = 60  # timesteps to plan for
        self.smooth_grad_a = 1.0  # how gradients fx, fu are smoothed out during a rollout a*fx_k + (1-a)*fx_(k-1)
        self.small_increment_a = 0.9
        self.du_alpha = 1e-3 # cost to ensure small perturbances du
        self.dx_alpha = 1e-3 # cost to ensure small perturbances dx

        #   convergence
        self.eps_cost = 1e-3  # cost tolerance [1e-2, 1e-4, 1e-8]        , HIGH
        self.eps_grad = (
            1e-5  # gradient tolerance 1e-5                  , MEDIUM (not used)
        )

        self.iter_max = 10  # max nr of backw/forward iter [50, 500]   , MEDIUM
        self.max_cost = 1e6  # max cost during rollout                  , MEDIUM

        #   line search
        self.alpha_init = 0.6  # initial alpha for the linesearch         , HIGH
        self.iter_line_search = 7  # Max nr of backtr line search [5,10, 20]  , LOW
        self.beta_1 = 1e-4  # lower bound for linesearch               , LOW
        self.beta_2 = 135.0  # upper bound for linesearch               , LOW

        #   regularization
        self.reg_init = 0.01  # Init value for reg Quu in backw pass 0   , LOW
        self.reg_max = 100  # Any further increase will saturate 1e8   , LOW
        self.reg_min = 1e-6  # Any reg below will round to 0   1e-8     , LOW
        self.reg_factor = 10  # how much reg is up/descaled (1,1.6,10)   , LOW


class ALConfigDefault(DotDict):
    def __init__(self) -> None:
        super().__init__()
        self._name = "AL"
        self.T = 60  # timesteps
        # Augmented Lagrangian hyperaparameters
        #   convergence
        self.al_max_iter = 4  # max nr of outer loop updates                          [10, 30, 100],             MEDIUM

        self.al_cmax = 1e-2  # convergence when max constraint violation < cmax      [1e-2, 1e-4, 1e-8],        HIGH
        self.al_eps_cost = None  # difference in cost between iterations for convergence [1e2, 1e-4, 1e-8],         HIGH (not used)
        self.al_eps_uncon = None  # cost tolerance for the iLQR to trigger al update(can require less optimality),[1e-1,1e-3,1e-8]   HIGH (not used)

        #   penalty
        self.al_mu_max = 1e7  # maximal penalty(high better conv), (low avoid ill)     1e8,                      LOW
        self.al_mu_init = 150.0  # initial penalty                                       [1e-4, 1, 100],            VERY HIGH
        self.al_psi = 1.4  # penalty scaling                                       (1, 10, 100],              MEDIUM
