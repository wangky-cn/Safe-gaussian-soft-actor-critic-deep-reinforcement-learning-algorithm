import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from TrackedTractorENV import TractorEnv
import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.layers = nn.ModuleList()
        input_dim = state_dim
        for h in hidden_width:
            self.layers.append(nn.Linear(input_dim, h))
            input_dim = h

        self.out = nn.Linear(hidden_width[-1], action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        a = self.max_action * torch.tanh(self.out(x))

        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()

        input_dim = state_dim + action_dim

        self.q1_layers = nn.ModuleList()
        self.q2_layers = nn.ModuleList()

        current_dim = input_dim
        for h in hidden_width:
            self.q1_layers.append(nn.Linear(current_dim, h))
            self.q2_layers.append(nn.Linear(current_dim, h))
            current_dim = h

        self.q1_out = nn.Linear(current_dim, 1)
        self.q2_out = nn.Linear(current_dim, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)

        q1 = s_a
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_out(q1)

        q2 = s_a
        for layer in self.q2_layers:
            q2 = F.relu(layer(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)

        q1 = s_a
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_out(q1)

        return q1

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(2e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            batch_size=120,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=5e-5,
            critic_lr=5e-5,
            hidden_sizes=[500, 400, 300],
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.hidden_sizes = hidden_sizes

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_sizes, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        print(self.actor)

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        print(self.critic)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.batch_size = batch_size

        self.total_it = 0
        self.q_loss = 0.0
        self.a_loss = 0.0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def learn(self, replay_buffer):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.q_loss = critic_loss.item()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.a_loss = actor_loss.item()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class LoggerVisualizer:
    def __init__(self, log_dir='./log', output_excel='training_logs.xlsx'):
        self.log_dir = log_dir
        self.output_excel = output_excel

    def save_to_excel(self):
        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()

        scalar_tags = ea.Tags()['scalars']

        training_data = {}
        loss_data = {}

        for tag in scalar_tags:
            scalar_values = ea.Scalars(tag)

            steps = [event.step for event in scalar_values]
            values = [event.value for event in scalar_values]

            if 'Episode' in tag or 'SoC' in tag or 'Fuel' in tag:
                tag = tag.replace(' ', '_')
                df = pd.DataFrame({"Step": steps, tag: values})
                df = df.drop_duplicates(subset='Step', keep='last')
                training_data[tag] = df
            elif 'loss' in tag or 'alpha' in tag or 'lam' in tag:
                tag = tag.replace(' ', '_')
                df = pd.DataFrame({"Step": steps, tag: values})
                df = df.drop_duplicates(subset='Step', keep='last')
                loss_data[tag] = df

        if training_data:
            training_merged = pd.concat([df.set_index('Step') for df in training_data.values()], axis=1).reset_index()

        if loss_data:
            loss_merged = pd.concat([df.set_index('Step') for df in loss_data.values()], axis=1).reset_index()

        with pd.ExcelWriter(self.output_excel) as writer:
            if training_data:
                training_merged.to_excel(writer, sheet_name="Training_Data", index=False)
            if loss_data:
                loss_merged.to_excel(writer, sheet_name="Loss_Data", index=False)

class Trainer:
    def __init__(self, env,agent,replay_buffer,max_train_steps,random_steps,device,expl_noise,summary_writer,frequency,visualizer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.max_train_steps = max_train_steps
        self.max_train_steps_ = 500
        self.random_steps = random_steps
        self.expl_noise = expl_noise

        self.summary_writer = summary_writer
        self.frequency = frequency
        self.visualizer = visualizer

        self.device = device
        self._generate_initial_soc()

    def _generate_initial_soc(self):
        random_list = np.arange(0.4, 0.81, 0.01)
        num = self.max_train_steps_ // len(random_list)
        random_list_ = np.tile(random_list, num)
        remaining_steps = self.max_train_steps_ - len(random_list_)
        if remaining_steps > 0:
            random_list_ = np.concatenate([random_list_, np.round(np.random.choice(random_list, remaining_steps), 2)])
        np.random.shuffle(random_list_)
        self.init_soc = random_list_

    def train(self):
        self.log_t = 0
        t_ = 0
        for t in range(int(self.max_train_steps)):
            self.env.init_soc = self.init_soc[t_]
            t_ += 1
            t_ = 0 if t_==len(self.init_soc) else t_
            s, done = self.env.reset(), False
            episode_reward = 0.0
            episode_cost = 0.0

            while not done:
                if t <= self.random_steps:
                    a = (self.env.action_space.high[0] - self.env.action_space.low[0]) * np.random.random(
                        self.env.action_space.shape) + self.env.action_space.low[0]
                else:
                    a = (self.agent.select_action(np.array(s)) + np.random.normal(0, self.env.action_space.high[0] * self.expl_noise, size=self.env.action_space.shape)).clip(-self.env.action_space.high[0], self.env.action_space.high[0])

                s_, r, c, done, _ = self.env.step(a)

                self.replay_buffer.store(s, a, r, s_, done)
                s = s_
                episode_reward += r
                episode_cost += c

                if t > self.random_steps:
                    self.agent.learn(self.replay_buffer)
                    if self.agent.total_it % self.frequency == 0:
                        self.summary_writer.add_scalar('actor loss', self.agent.a_loss, self.log_t)
                        self.summary_writer.add_scalar('critic loss', self.agent.q_loss, self.log_t)
                        self.log_t += 1

                if done:
                    self.summary_writer.add_scalar('Episode Reward', episode_reward, t)
                    self.summary_writer.add_scalar('Episode Cost', episode_cost, t)
                    self.summary_writer.add_scalar('SoC0', float(self.env.init_soc), t)
                    self.summary_writer.add_scalar('SoC', s[2], t)

                    if t % 10 == 0:
                        print("episode:", t,
                          "| Init SoC: %.3f" % self.env.init_soc,
                          "| Final SoC: %.3f" % s[2],
                          "| reward: %.3f" % episode_reward,
                          "| cost: %.3f" % episode_cost,
                          )

        self.summary_writer.close()
        self.visualizer.save_to_excel()


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TractorEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    max_episode_steps = env.model.TotalStep

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = f'./log/seed{seed}'
    os.makedirs(log_dir, exist_ok=True)
    output_excel = f'./log2excel/training_logs_seed{seed}.xlsx'
    frequency = 500
    summary_writer = SummaryWriter(log_dir)
    visualizer = LoggerVisualizer(log_dir, output_excel)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": device,
        "batch_size": 120,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.15,
        "noise_clip": 0.3,
        "policy_freq": 2,
        "actor_lr": 5e-5,
        "critic_lr": 5e-5,
        "hidden_sizes": [500, 400, 300],
    }

    agent = TD3(**kwargs)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    max_train_steps = 1500
    total_steps = 0

    random_steps = 5
    expl_noise = 0.1

    trainer = Trainer(env, agent, replay_buffer, max_train_steps, random_steps, device, expl_noise, summary_writer,frequency,visualizer)
    trainer.train()




