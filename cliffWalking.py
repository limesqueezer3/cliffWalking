import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Create the CliffWalking environment
# https://gymnasium.farama.org/environments/toy_text/cliff_walking/
env = gym.make('CliffWalking-v0', render_mode="rgb_array")

# Reset the environment to start
obs, info = env.reset()

for _ in range(1):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    # Render the environment and get the image array
    image = env.render()
    print("Initial Observation:", obs)
    # Plot the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the image
    if done:
        break

### action = env.action_space.sample()  # Random action
# obs, reward, done, truncated, info = env.step(3)
# # Render the environment and get the image array
# image = env.render()
# print("Initial Observation:", obs)
# # Plot the image using matplotlib
# plt.imshow(image)
# plt.axis('off')  # Turn off the axis
# plt.show()  # Display the image

# env.close()


# unwrapped_env = env.unwrapped

# ret = unwrapped_env.P[43][0]
# print(ret)

# env.observation_space.n

#using fixed random policy
def policyEvaluation (theta, gamma, env) :
    unwrapped_env = env.unwrapped
    #initialize V
    V = np.zeros(env.observation_space.n) 
    delta = theta+1
    while delta > theta :
        delta = 0
        Vnew = np.zeros(env.observation_space.n)
        for s in range(env.observation_space.n) :
            v = V[s]
            if s == 47 :
                Vnew[s] = 0
            else :
                for action in range(4) :
                    #get the next reward/state etc, prob is just 1 
                    prob, next_state, reward, done = unwrapped_env.P[s][action][0]  
                    Vnew[s] += 0.25*(reward + gamma * V[next_state])
            if v < Vnew[s] : 
                delta = max(delta, abs(Vnew[s] - v)) 
            else :
                delta = max(delta, abs(v - Vnew[s])) 
        V = Vnew
    return V


# Define parameters
theta = 0.1  # Small threshold for convergence
gamma = 0.7   # Discount factor

# Run policy evaluation
V = policyEvaluation(theta, gamma, env)

# Print the value function
print("Value Function V(s):")
print(V.reshape(4, 12))  # Reshape to match the grid world layout

# Visualize the value function
plt.imshow(V.reshape(4, 12), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Value Function V(s)")
plt.show()
        
        
        
    