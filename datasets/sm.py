import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial
from statistics import mean

class SelkovModelDataset(Dataset):

    def __init__(self, num_traj_per_env, time_horizon, params, dt, batch_t=10, method='RK45', group='train'):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.num_env = len(params)
        self.len = num_traj_per_env * self.num_env
        self.time_horizon = float(time_horizon)       # total time
        self.dt = dt
        self.batch_t = batch_t

        self.params_eq = params
        self.test = group == 'test'
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.method = method
        self.indices = [list(range(env * num_traj_per_env, (env + 1) * num_traj_per_env)) for env in range(self.num_env)]

    def _f(self, t, x, env=0):
        a, b = self.params_eq[env]

        x, y = x
        dx = -x + a*y + (x**2)*y
        dy = b - a*y - (x**2)*y
        return np.array([dx, dy])

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max-index)
        return np.random.uniform(0, 3, 2)

    def __getitem__(self, index):
        env       = index // self.num_traj_per_env
        env_index = index %  self.num_traj_per_env
        t = torch.arange(0, self.time_horizon, self.dt).float()
        t0 = torch.randint(t.size(0) - self.batch_t + 1, (1,)).item()
        if self.buffer.get(index) is None:
            y0 = self._get_init_cond(env_index)

            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=y0, method=self.method, t_eval=np.arange(0., self.time_horizon, self.dt))
            res = torch.from_numpy(res.y).float()
            self.buffer[index] = res.numpy()
            return {
                'state'   : res,
                't'       : t,
                'env'     : env,
            }
        else:
            return {
                'state'   : torch.from_numpy(self.buffer[index]),
                't'       : t,
                'env'     : env,
            }

    def __len__(self):
        return self.len