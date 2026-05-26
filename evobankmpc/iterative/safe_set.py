
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor


def get_subsample_inds(length, subsample_num):
    if subsample_num is None:
        return np.arange(length)
    if subsample_num > length:
        subsample_num = length
    return np.random.permutation(length)[:subsample_num]


class SafeSet:


    def __init__(self, config, logline, track=None):

        self.logline = logline
        self.config = config
        self.track = track
        self.value_dim = getattr(config, 'value_dim', 4)           # [s, ey, epsi, v]
        self.ss_hull_num_reduction = getattr(config, 'ss_hull_num_reduction', 10)
        self.ss_hull_precompile_len = getattr(config, 'ss_hull_precompile_len', 1000)
        self.n_samples = getattr(config, 'n_samples', 512)       
        self.n_steps = getattr(config, 'n_steps', 10)            
        self.ss_arr_max_len = getattr(config, 'ss_arr_max_len', 200000)
        self.ss_select_max_len = getattr(config, 'ss_select_max_len', 1000)
        self.ss_loop_extend_sec = getattr(config, 'ss_loop_extend_sec', 2.0)
        self.sim_time_step = getattr(config, 'sim_time_step', 0.02)
        self.ss_ref_step_interval = getattr(config, 'ss_ref_step_interval', 3.0)
        self.init_vel = getattr(config, 'init_vel', 5.0)
        self.half_width = getattr(config, 'half_width', 4.0)

        self.s_frame_max = getattr(config, 's_frame_max', 0.0)


        self.precompile_range = np.arange(self.ss_hull_num_reduction,
                                          self.ss_hull_precompile_len,
                                          self.ss_hull_num_reduction)

        self.lap_record_frenet = []     
        self.lap_record_carti = []      
        self.safe_set_frenet = []       
        self.safe_set_carti = []        
        self.ss_arr_frenet = None       
        self.ss_arr_carti = None        

        self.xSS_hull_equations = None
        self.len_xSS_hull = 0
        self.compiled = {}           
        self.compiled2 = {}

    def add_lap(self, time_lap, frenet_lap=None, carti_lap=None):
        values = np.asarray(time_lap).reshape(-1, 1)
        if frenet_lap is not None:
            self.lap_record_frenet.append(np.concatenate([np.asarray(frenet_lap), values], axis=1))
            if carti_lap is not None:
                self.lap_record_carti.append(np.concatenate([np.asarray(carti_lap), values], axis=1))
        else:
            self.lap_record_carti.append(np.concatenate([np.asarray(carti_lap), values], axis=1))

    def get_new_safe_set(self):
        new_safe_set = False
        if len(self.lap_record_frenet) > 1:
            max_speed = np.max(self.lap_record_frenet[-1][:, 3]) 
            extend_dist = max_speed * self.ss_loop_extend_sec * self.n_steps * self.sim_time_step
            extend_inds = np.where(self.lap_record_frenet[-1][:, 0] < extend_dist)[0]
            if len(extend_inds) > 0:
                extended_lap = self.lap_record_frenet[-1][extend_inds].copy()
                extended_lap[:, 0] += self.s_frame_max
                extended_lap[:len(self.lap_record_frenet[-2]), -1] -= np.max(extended_lap[:, -1])
                combined = np.concatenate([self.lap_record_frenet[-2], extended_lap], axis=0)
                self.safe_set_frenet.append(combined)
                if len(self.lap_record_carti) > 1:
                    extended_lap_xy = self.lap_record_carti[-1][extend_inds].copy()
                    if self.track is not None:
                        pass
                    combined_xy = np.concatenate([self.lap_record_carti[-2], extended_lap_xy], axis=0)
                    combined_xy[:, -1] = combined[:, -1]   
                    self.safe_set_carti.append(combined_xy)

                self.lap_record_frenet.pop(0)
                if self.lap_record_carti:
                    self.lap_record_carti.pop(0)
                new_safe_set = True

        elif len(self.lap_record_carti) >= 1:
            lap_xy = self.lap_record_carti[-1]
            lap_xy[:, -1] -= np.max(lap_xy[:, -1])   
            self.safe_set_carti.append(lap_xy)
            if self.lap_record_frenet:
                lap_f = self.lap_record_frenet[-1]
                lap_f[:, -1] -= np.max(lap_f[:, -1])
                self.safe_set_frenet.append(lap_f)
            new_safe_set = True

        ss_size = getattr(self.config, 'ss_size', 4)
        while len(self.safe_set_carti) >= ss_size + 1:
            self.safe_set_carti.pop(0)
        while len(self.safe_set_frenet) >= ss_size + 1:
            self.safe_set_frenet.pop(0)

        return new_safe_set

    def update_ss_arr(self):
        if len(self.safe_set_frenet) > 0:
            self.ss_arr_frenet = np.concatenate(self.safe_set_frenet)
        if len(self.safe_set_carti) > 0:
            self.ss_arr_carti = np.concatenate(self.safe_set_carti)

        if self.ss_arr_frenet is not None and self.ss_arr_frenet.shape[0] > self.ss_arr_max_len:
            subsample_inds = get_subsample_inds(self.ss_arr_frenet.shape[0], self.ss_arr_max_len)
            self.ss_arr_frenet = self.ss_arr_frenet[subsample_inds]
            if self.ss_arr_carti is not None:
                self.ss_arr_carti = self.ss_arr_carti[subsample_inds]

    def update_convex_hull(self, ref_ss_states):
        if ref_ss_states.shape[0] < 3:
            return
        hull = ConvexHull(ref_ss_states[:, :self.value_dim])
        equations = hull.equations   # (n_facets, value_dim+1)
        if equations.shape[0] % self.ss_hull_num_reduction != 0:
            equations = equations[:-(equations.shape[0] % self.ss_hull_num_reduction)]
        self.xSS_hull_equations = jnp.array(equations)
        self.len_xSS_hull = self.xSS_hull_equations.shape[0]

    def get_ss_dist(self, states):
        if self.xSS_hull_equations is None:
            return jnp.zeros(states.shape[0])
        states_proj = states[:, :self.value_dim]
        try:
            if self.n_samples == states.shape[0]:
                dist = self.compiled[self.len_xSS_hull](states_proj, self.xSS_hull_equations)
            else:
                dist = self.compiled2[self.len_xSS_hull](states_proj, self.xSS_hull_equations)
        except (KeyError, AttributeError):
            dist = self.vmap_sshull_reward(states_proj, self.xSS_hull_equations)
        return dist

    def find_ss_inrange(self, terminal_states, ss_arr, env_state):
        s_vals = terminal_states[:, 0]
        s_max = np.max(s_vals)
        s_min = np.min(s_vals)
        speed = max(env_state[3], self.init_vel)
        extend = speed * (self.ss_ref_step_interval + 1) * self.sim_time_step
        s_low = s_min - extend
        s_high = s_max + extend
        ss_s = ss_arr[:, 0]
        inds = np.where((ss_s > s_low) & (ss_s < s_high))[0]
        if len(inds) > self.ss_select_max_len:
            inds = get_subsample_inds(len(inds), self.ss_select_max_len)
        return (inds,)

    def find_ss_inrange_zt(self, zt, ss_arr):
        speed = max(zt[3], self.init_vel)
        s_next = zt[0] + speed * self.sim_time_step
        ss_s = ss_arr[:, 0]
        diffs = np.abs(ss_s - s_next)
        closest = np.argpartition(diffs, self.ss_select_max_len)[:self.ss_select_max_len]
        return (closest,)

    @partial(jax.jit, static_argnums=(0,))
    def sshull_reward(self, state, hull_eq):
        dist = jnp.max(jnp.dot(hull_eq[:, :-1], state) + hull_eq[:, -1])
        dist = dist * (dist > 0)
        return dist

    @partial(jax.jit, static_argnums=(0,))
    def vmap_sshull_reward(self, states, hull_eq):
        return jax.vmap(self.sshull_reward, in_axes=(0, None))(states, hull_eq)

    def ss_hull_precompile(self, lambs_len):
        if self.logline:
            self.logline('Precompiling jit for ss hull')
        dummy_input1 = jnp.ones((self.n_samples, self.value_dim))
        dummy_input2 = jnp.ones((lambs_len, self.value_dim))
        dummy_equations = [jnp.ones((ind, self.value_dim + 1)) for ind in self.precompile_range]
        compiled = {}
        compiled2 = {}
        with ThreadPoolExecutor() as pool:
            for eq in dummy_equations:
                fn1 = jax.jit(self.vmap_sshull_reward, backend='gpu').lower(dummy_input1, eq)
                fn2 = jax.jit(self.vmap_sshull_reward, backend='gpu').lower(dummy_input2, eq)
                compiled[eq.shape[0]] = pool.submit(fn1.compile)
                compiled2[eq.shape[0]] = pool.submit(fn2.compile)
            self.compiled = {s: f.result() for s, f in compiled.items()}
            self.compiled2 = {s: f.result() for s, f in compiled2.items()}
        for eq in dummy_equations:
            self.compiled[eq.shape[0]](dummy_input1, eq)
            self.compiled2[eq.shape[0]](dummy_input2, eq)