import numpy as np
import pandas as pd
import gym
from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def trainLunarLander():
    # Create the environment
    env = make_vec_env('LunarLander-v2', n_envs=16)
    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1)
    # Train it for 500,000 timesteps
    model.learn(total_timesteps=3500000)
    # Save the model
    model_name = "ppo-LunarLander-v3"
    model.save(model_name)
    return model

def evalLunarlander(model):
    eval_env = gym.make("LunarLander-v2")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=False)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

def createDataset(model, numMuestra):
    arr=[]
    directory = './video'
    env = gym.make("LunarLander-v2")
    episodio=1
    for i in range(0,numMuestra):
        obs = env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            puntos=[]
            puntos.append(episodio)
            puntos.append(obs[0]) #x
            puntos.append(obs[1]) #y
            puntos.append(obs[2]) #vx
            puntos.append(obs[3]) #vy
            puntos.append(obs[4]) #theta
            puntos.append(obs[5]) #vtheta
            puntos.append(obs[6]) #legx
            puntos.append(obs[7]) #legy
            puntos.append(action[()])
            arr.append(puntos)    
        episodio=episodio+1
    return arr

def datasetToFile(arr, nameFile):
    nparr=np.asarray(arr)
    DF = pd.DataFrame(nparr)
    DF.to_csv(nameFile)


if __name__ == '__main__':
    #model=trainLunarLander()
    env = make_vec_env('LunarLander-v2', n_envs=16)
    ppo = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        verbose=1)
    model=ppo.load('ppo-LunarLander-v3')
    evalLunarlander(model)
    arr=createDataset(model, 50000)
    datasetToFile(arr, 'Lulander50000.dat')