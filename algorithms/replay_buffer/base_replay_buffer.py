import abc


class BaseReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def __init__():
        pass
    
    @abc.abstractmethod
    def reset_buffer(self):
        pass
    
    @abc.abstractmethod
    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        pass
    
    @abc.abstractmethod
    def store_last_value(self, episode_step, v_n):
        pass