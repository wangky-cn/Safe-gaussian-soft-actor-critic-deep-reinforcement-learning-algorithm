import matlab.engine
import gym
import numpy as np
from gym import spaces

class Env(gym.Env):
    def __init__(self):
        self.engine = matlab.engine.start_matlab()
        self.engine.dllModel_RAM_ROM(nargout=0)
        self.done_t = 1145
        self.control_step = 1

        # self.action_space = spaces.Box(low=-2.5, high=2.5, shape=(1,), dtype=np.float64)
        # self.observation_space = spaces.Box(
        #     low=np.array([0.0, 0.0, 0.0, 0.0]),
        #     high=np.array([1.0, 1.0, 1.0, 1.0]),
        #     dtype=np.float64
        # )

    def reset(self):
        self.engine.Init_Env(nargout=0)
        self.engine.Get_State(nargout=0)
        self.done = False
        state = self.engine.eval('APUPCMD_4DRL', nargout=1)
        self.t = self.control_step
        return state, self.done

    def step(self, action):
        self.engine.Step_Env(action, self.t, nargout=0)
        self.engine.Get_State(nargout=0)
        state = self.engine.eval('APUPCMD_4DRL', nargout=1)
        reward = self.calculate_reward(state)
        self.done = True if self.t == self.done_t else False
        self.t += self.control_step
        return state, reward, self.done, {}

    def calculate_reward(self,state):
        return -10

    def Stop(self):
        self.engine.Stop_Env(nargout=0)


if __name__ == '__main__':
    env = Env()
    act_list = []
    dll_fbk_list = []

    s, done = env.reset()

    act_list.append(0)
    dll_fbk_list.append(s)
    t = 0

    while not done:
        a = t
        s, r, done, _ = env.step(a)
        act_list.append(a)
        dll_fbk_list.append(s)
        t += 1
    env.Stop()

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    x = range(len(act_list))
    y1 = act_list
    y2 = dll_fbk_list
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, linestyle='-', label='Act List Curve')
    plt.plot(x, y2, linestyle='-', label='Simulink Feedback Curve')
    plt.title('Comparison of Action and DLL Feedback', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    annotation_text = (
        "Simulink is twice as large as Act\n"
        "and is consistent with the Simulink model,\n"
        "indicating that the environment is correct."
    )
    plt.annotate(
        annotation_text,
        xy=(0.2, 0.5), xycoords='axes fraction',
        fontsize=10, color='blue',
        bbox=dict(boxstyle="round", fc="w", ec="blue", lw=1, alpha=0.5)
    )
    plt.legend()
    plt.grid(True)
    plt.show()



