"""
Akhilesh Khope
PhD Candidate 
Electrical and Computer Engineering
UC Santa Barbara

https://github.com/aspk/Tensorflow_Deep_RL/

This is an implementation of actor critic algorithm with parameter space exploration for deep reenforcement learning

Parameter Space Noise for Exploration
https://arxiv.org/abs/1706.01905


""" 

import gym
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
gamename = 'CartPole-v0'
env = gym.make(gamename)
obs_space = env.observation_space.shape[0]
ac_space = env.action_space.n

def linear(x,in_dim,out_dim,scope_name):
    with tf.variable_scope(scope_name):
        w = tf.get_variable('w',[in_dim,out_dim],initializer = tf.random_normal_initializer())
        b = tf.get_variable('b',[out_dim],initializer = tf.random_normal_initializer())
        
        return tf.matmul(x,w)+b      

episode_lengths = []
rewards_global= []
def run_episode(env):
	sess = tf.get_default_session()
	total_rew = 0
	act = []
	obs = []
	re  = []
	observation = env.reset()        
	action = env.action_space.sample()
	for t in range(1000):    
		#env.render()
		observation, reward, done, info = env.step(action)
		action = det_policy(observation)
		act_temp = np.zeros(env.action_space.n,np.float32)
		act_temp[action] = 1
		act.append(act_temp)
		obs.append(observation)
		re.append(reward)
		total_rew+=reward
		if done:
			#print("Episode finished after {} timesteps".format(t+1))
			break
	rewards_global.append(total_rew)
	return {'action' : np.array(act),
			'obs'    : np.array(obs),
			'reward' : np.array(re)
	}
def discount(x, gamma):
	"""
	Given vector x, computes a vector y such that
	y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
	"""
	out = scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
	return out
def act_policy(observation):
	sess = tf.get_default_session()
	action = sess.run(y,feed_dict={x:[observation],keep_prob:1})
	return np.random.choice([0,1],p = action.ravel())
def det_policy(observation):
    sess = tf.get_default_session()
    action = sess.run(y,feed_dict={x:[observation],keep_prob:1})
    return np.argmax(action)
def get_trajs(env,timesteps,gamma,noise_std,old_var):
	sess = tf.get_default_session()
	max_len = []
	count = 0
	run_stat = {}
	while count<timesteps:
		noisy_vars(noise_std)
		if count==0:
			run_stat = run_episode(env)
			run_stat['reward'] = discount(run_stat['reward'],gamma)
			count+=len(run_stat['reward'])
			max_len.append(len(run_stat['reward']))
		else:
			temp = run_episode(env)
			temp['reward'] = discount(temp['reward'],gamma)
			run_stat['action']=np.concatenate([run_stat['action'],temp['action']])
			run_stat['obs']=np.concatenate([run_stat['obs'],temp['obs']])
			run_stat['reward']=np.concatenate([run_stat['reward'],temp['reward']])
			count+=len(temp['reward'])
			max_len.append(len(temp['reward']))
		episode_lengths.append(max_len)
		reset_vars(old_var)
		#print(count)
	print('max episode length  is  {}'.format(max(max_len)))  
	return run_stat


#network
graph = tf.Graph()
with graph.as_default():
    x  = tf.placeholder(tf.float32,shape = [None,obs_space])
    adv = tf.placeholder(tf.float32,shape = [None])
    rew = tf.placeholder(tf.float32,shape = [None,1])
    ac = tf.placeholder(tf.float32,shape = [None,ac_space])
    keep_prob = tf.placeholder(tf.float32)

    x1 = linear(x,in_dim = obs_space,out_dim = ac_space,scope_name = 'l1')
    #x1 = tf.nn.dropout(x1, keep_prob)
    #policy network
    #x2 = linear(x1,in_dim = 4,out_dim = ac_space,scope_name = 'l2') 
    y  = tf.nn.softmax(x1)
    #value network
    #x3 = tf.nn.dropout(x1, keep_prob)
    y_v = linear(x,in_dim = obs_space,out_dim = 1,scope_name = 'l3')

    #value loss
    loss_v = tf.nn.l2_loss(y_v-rew)
    # policy loss
    log_prob_tf = tf.log(y)
    loss1 = tf.reduce_sum(log_prob_tf * ac, [1])
    loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * ac, [1])*adv) + loss_v
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init = tf.global_variables_initializer()

def train(run_stat,numsteps,batch_size,dropout = 1):
	sess = tf.get_default_session()
	value = sess.run(y_v,feed_dict={x:run_stat['obs'][:numsteps], keep_prob:1.0})
	advantage = run_stat['reward'][:numsteps].reshape(numsteps,1)-value
	advantage = advantage.reshape(len(advantage))
	#print('training value')
	for i in range(numsteps//batch_size):
		batchobs = run_stat['obs'][i*batch_size:(i+1)*batch_size]
		batchrew = run_stat['reward'][i*batch_size:(i+1)*batch_size]
		batchrew = batchrew.reshape(len(batchrew),1)
		batchadv = advantage[i*batch_size:(i+1)*batch_size]
		batchac = run_stat['action'][i*batch_size:(i+1)*batch_size]
		#print("optimizing")
		sess.run(optimizer,feed_dict = {x: batchobs, rew: batchrew, adv: batchadv, ac:batchac, keep_prob:dropout})	#keep_prob<1
	return 

def noisy_vars(noise_std = 1):
	sess = tf.get_default_session()
	var_names = tf.global_variables()
	old_var = sess.run(var_names)
	var_shapes = [i.shape for i in old_var]
	new_var = [i+np.random.normal(0,noise_std,size = j) for i,j in zip(old_var,var_shapes)]
	# setting new values
	for i,j in zip(var_names,new_var):
		sess.run(i.assign(j))
	return 

def reset_vars(old_var):
	sess = tf.get_default_session()
	var_shapes = [i.shape for i in old_var]
	# setting old values
	for i,j in zip(var_names,old_var):
		sess.run(i.assign(j))
	return


episode_lengths = []
rewards_global = []
with tf.Session(graph=graph) as sess:
	init.run()
	numsteps = 1000
	gamma = 0.99
	batch_size = 32
	epochs = 100
	var_names = tf.global_variables()
	for i in range(epochs):
		if i%10==0:
			print("{}:{}".format(i,epochs))
		
		old_var = sess.run(var_names)
		var_shapes = [i.shape for i in old_var]

		run_stat = get_trajs(env = env,timesteps=numsteps,gamma=gamma,noise_std = 1,old_var = old_var)

		# resetting old values
		# reset_vars(old_var)

		#train_network
		#print('train')
		train(run_stat,numsteps,batch_size,dropout = 1)

def mvavg(a,width = 300):
	mvavg = np.zeros(len(a))
	for i in range(len(a)-width+1):
		mvavg[i] = np.mean(a[i:i+width])
	return mvavg
width = 300
plt.figure(figsize=(20,6))
plt.suptitle(gamename, fontsize=16)

epi_lengths = [item for sublist in episode_lengths for item in sublist]
epi_avg = mvavg(epi_lengths[:],width = width)[:-(width-1)]
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplot(1,2,1)
plt.plot(np.arange(0,len(epi_lengths)),epi_lengths,alpha = 0.2,color = 'orange')
epi_avg = mvavg(epi_lengths[:],width = width)[:-(width-1)]
plt.plot(np.arange(len(epi_avg)),epi_avg,color = 'orange',linewidth = 2)
plt.xlabel('Number of evaluations', fontsize=20)
plt.ylabel('Episode Length', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(1,2,2)
width = 20
rew_avg = mvavg(rewards_global[:],width = width)[:-(width-1)]
plt.plot(np.arange(0,len(rewards_global)),rewards_global,alpha = 0.2,color = 'orange')
plt.plot(np.arange(len(rew_avg)),rew_avg,color = 'orange',linewidth = 2)
plt.xlabel('Number of evaluations', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(gamename + 'episode_len_reward.png',bbox_inches='tight',dpi = 300)
plt.show()
