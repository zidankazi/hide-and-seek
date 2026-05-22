import torch
from env import HideAndSeekEnv
from ppo_continuous import ActorCritic

# Load the trained policy and watch it play

env = HideAndSeekEnv(render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

ac = ActorCritic(obs_dim, act_dim)
ac.load_state_dict(torch.load("policy.pt"))
ac.eval()

obs, _ = env.reset()
total_reward = 0
episode = 0

while True:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = ac(obs_t) # Use the mean of the policy instead of sampling, for deterministic behavior
    action = mean.squeeze(0).numpy()

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()

    if terminated or truncated:
        episode += 1
        print(f"Episode {episode} | Return: {total_reward:.1f} | Reached goal: {terminated}")
        total_reward = 0
        obs, _ = env.reset()
