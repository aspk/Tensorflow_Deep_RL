Tensorflow_Deep_RL
**********

An implementation of multiple algorithms used for training agents on gym environment.


Parameter Space Noise for Exploration 
=====================================

Parameter noise ius used for exploration in this method. Standard deviation of noise is constant.
 
https://arxiv.org/abs/1706.01905

Filename: param_noise_mlp.py

Dependencies
--------------
Python 3.6 with Anaconda_, Tensorflow_, OpenAI Gym_ required.

.. _Tensorflow : https://www.tensorflow.org/install/
.. _Anaconda : https://www.anaconda.com/download/#macos
.. _Gym : https://github.com/openai/gym

Usage
--------

.. code:: shell
    
    python param_noise_mlp.py
   
Graph
-----------

.. code:: python

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
        
Noisy parameters
----------------

Noise is added to the parameters while collection rollouts for exploration and deterministic actions are used instead of probabilistic as in the case of actor critic methods.

.. code:: python

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
        
Train
------------
.. code:: python

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
            sess.run(optimizer,feed_dict = {x: batchobs, rew: batchrew, adv: batchadv, ac:batchac, keep_prob:dropout})	                   #keep_prob<1
        return
        
