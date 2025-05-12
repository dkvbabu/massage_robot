from massage_robot.env import MassageEnv
import time

if __name__ == "__main__":
    env = MassageEnv(gui=True)
    obs = env.reset()
    print("Environment ready. Press Ctrl+C to exit.")
    try:
        while True:
            obs = env.step([0.0]*7)
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("Exiting viewer.")
