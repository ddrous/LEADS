import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial


class BrusselatorDataset(Dataset):
    def __init__(self, num_traj_per_env, size, time_horizon, params, dt_eval, n_block, dx=2., random_influence=0.2,
                 buffer=dict(), method='RK45', group='train'):
        super().__init__()
        dt = dt_eval
        t_horizon = time_horizon
        n_data_per_env = num_traj_per_env
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.size = int(size)  # size of the 2D grid
        self.dx = dx  # space step discretized domain [-1, 1]
        self.time_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt)  # number of iterations
        self.random_influence = random_influence
        self.dt_eval = dt
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = buffer
        self.method = method
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.n_block = n_block

    def _laplacian2D(self, a):
        # a_nn | a_nz | a_np
        # a_zn | a    | a_zp
        # a_pn | a_pz | a_pp
        a_zz = a

        a_nz = np.roll(a_zz, (+1, 0), (0, 1))
        a_pz = np.roll(a_zz, (-1, 0), (0, 1))
        a_zn = np.roll(a_zz, (0, +1), (0, 1))
        a_zp = np.roll(a_zz, (0, -1), (0, 1))

        a_nn = np.roll(a_zz, (+1, +1), (0, 1))
        a_np = np.roll(a_zz, (+1, -1), (0, 1))
        a_pn = np.roll(a_zz, (-1, +1), (0, 1))
        a_pp = np.roll(a_zz, (-1, -1), (0, 1))

        return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (self.dx ** 2)

    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv, env=0):
        U, V = self._vec_to_mat(uv)
        deltaU = self._laplacian2D(U)
        deltaV = self._laplacian2D(V)
        dUdt = (self.params_eq[env]['Du'] * deltaU + self.params_eq[env]['A'] - (self.params_eq[env]['B']+1)* U + (U**2) * V)
        dVdt = (self.params_eq[env]['Dv'] * deltaV + self.params_eq[env]['B'] * U - (U**2) * V)
        duvdt = self._mat_to_vec(dUdt, dVdt)
        return duvdt

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        size = (self.size, self.size)
        A = np.random.uniform(0.5, 2.0)
        B = np.random.uniform(1.25, 5.0)

        U = A * np.ones(size)
        V = (B/A) * np.ones(size) + 0.1 * np.random.randn(self.size, self.size)

        return U, V

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.time_horizon, self.dt_eval).float()
        if self.buffer.get(f'{env},{env_index}') is None:
        # if True:          ## TODO grey-scott has a buffer problem !
            print(f'generating {env},{env_index}')
            uv_0 = self._mat_to_vec(*self._get_init_cond(env_index))
            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=uv_0, method=self.method,
                            t_eval=np.arange(0., self.time_horizon, self.dt_eval))
            res_uv = res.y
            u, v = [], []
            for i in range(self.n):
                res_U, res_V = self._vec_to_mat(res_uv[:, i])
                u.append(torch.from_numpy(res_U).unsqueeze(0))
                v.append(torch.from_numpy(res_V).unsqueeze(0))
            u = torch.stack(u, dim=1)
            v = torch.stack(v, dim=1)
            state = torch.cat([u, v], dim=0).float()
            self.buffer[f'{env},{env_index}'] = {'state': state.numpy()}
            return {'state': state, 't': t, 'env': env, 'index': index}
        else:
            buf = self.buffer[f'{env},{env_index}']
            return {'state': torch.from_numpy(buf['state']), 't': t, 'env': env, 'index': index}

    def __len__(self):
        return self.len
