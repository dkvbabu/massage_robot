def test_simulation_runs():
    from massage_robot.env import MassageEnv
    env = MassageEnv(render=False)
    for _ in range(10):
        action = env.get_action()
        env.step(action)
    env.close()

def test_dqn_tensorboard():
    from massage_robot.dqn import train_dqn
    train_dqn()

def test_ppo_tensorboard():
    from massage_robot.ppo import train_ppo
    train_ppo() 