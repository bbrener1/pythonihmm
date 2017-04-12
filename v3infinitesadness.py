#!/usr/bin/env python
from __future__ import division
import sys
import numpy as np
import random
import copy
from scipy.special import expit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.animation import ImageMagickFileWriter
from matplotlib.animation import FuncAnimation

def dirichlet_sample(alpha):
    dirichlet_vector = np.random.gamma(alpha,1)
    return np.divide(dirichlet_vector/np.sum(dirichlet_vector))

class hiddenModel:

    def __init__(self,observations, alpha=1,gamma=1, alpha_a = 4, alpha_b = 2, gamma_a = 3, gamma_b = 6):

        self.observations = observations
        self.observation_set = set(self.observations)
        self.observation_index_list = list(self.observation_set)
        self.observation_array = np.asarray(map(lambda x: self.observation_index_list.index(x), observations))

        self.state_list = []
        self.state_assignments = []
        self.state_array = np.zeros(self.observation_array.shape,dtype=int)

        self.transition_matrix = np.zeros((0,0))
        self.emission_matrix = np.zeros((0,len(self.observation_set)))
        self.emission_priors =  np.ones(0)


        self.alpha0 = alpha

        self.alpha0_a = alpha_a
        self.alpha0_b = alpha_b


        self.beta = None

        self.gamma = gamma
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b

        self.hyper_resampling_num = 20

    def initialize_test(self):
        self.add_state()
        self.add_state()
        self.add_state()
        self.add_state()

        self.remove_state(1)
        self.state_array = np.asarray([0,1,0,1,0,1])
        for state in self.state_array:
            self.state_assignments.append(self.state_list[state])
        self.count_state_transitions()
        self.count_state_emissions()
        self.beta = np.ones(len(self.state_list)+1,dtype=float) / np.sum(len(self.state_list)+1)
        # print "DEBUG COUNTS"
        # print self.transition_matrix
        # print self.emission_matrix
        # print self.observation_array


    def initialize_random(self,states=10):
        for i in range(states+1):
            self.add_state()
        self.remove_state(0)
        for i, observation in enumerate(self.observation_array):
            self.state_assignments.append(random.choice(self.state_list))
            self.state_array[i] = self.state_assignments[i].index
        self.count_state_transitions()
        self.count_state_emissions()

        # print "DEBUG COUNTS"
        # print self.transition_matrix
        # print self.emission_matrix
        # print self.observation_array
        # print self.state_array




    def add_state(self):
        self.state_list.append(hiddenState(len(self.state_list),self))
        self.transition_matrix = np.append(self.transition_matrix,np.zeros((1,self.transition_matrix.shape[1])),axis=0)
        self.transition_matrix = np.append(self.transition_matrix,np.zeros((self.transition_matrix.shape[0],1)),axis=1)
        self.emission_matrix = np.append(self.emission_matrix,np.zeros((1,self.emission_matrix.shape[1])),axis=0)
        if self.beta != None:
            new_beta = np.random.beta(1, self.gamma)
            last_beta = self.beta[-1]
            self.beta = np.append(self.beta,np.zeros(1))
            self.beta[-2] = new_beta * last_beta
            self.beta[-1] = (1.-new_beta) * last_beta
        else:
            self.beta = np.ones(1,dtype=float)
            new_beta = np.random.beta(1, self.gamma)
            last_beta = self.beta[-1]
            self.beta = np.append(self.beta,np.zeros(1))
            self.beta[-2] = new_beta * last_beta
            self.beta[-1] = (1.-new_beta) * last_beta

        self.emission_priors = np.append(self.emission_priors,np.ones(1))

    def remove_state(self,r_state):
        # print "REMOVE STATE DEBUG"
        # print len(self.state_list)
        # print self.transition_matrix.shape
        del(self.state_list[r_state])
        for i, item in enumerate(self.state_list):
            item.index = i
        for i, state in enumerate(self.state_assignments):
            self.state_array[i] = state.index
        self.transition_matrix = np.delete(self.transition_matrix, r_state, axis=0)
        self.transition_matrix = np.delete(self.transition_matrix, r_state, axis=1)
        print len(self.state_list)
        print self.transition_matrix.shape
        print self.alpha0
        print self.gamma
        self.emission_matrix = np.delete(self.emission_matrix, r_state, axis=0)
        self.beta = np.delete(self.beta, r_state, axis=0)
        self.emission_priors = np.delete(self.emission_priors, r_state, axis=0)
        self.count_state_transitions()
        self.count_state_emissions()


    def count_state_transitions(self):
        # print "TRANSITION COUNT DEBUG"
        self.transition_matrix = np.zeros(self.transition_matrix.shape)
        self.transition_matrix[1,1] = 1
        for i in range(1,len(self.state_array)):
            self.transition_matrix[self.state_array[i-1],self.state_array[i]] += 1


    def count_state_emissions(self):
        for state in self.state_list:
            for i, observation in enumerate(self.observation_index_list):
                # print self.state_array
                # print self.observation_array
                # print observation
                # print i
                # print i==observation
                state_occurrences = self.state_array == state.index
                observation_occurences = self.observation_array == i
                state.emission_vector[i] = np.sum(np.logical_and(state_occurrences,observation_occurences))
                self.emission_matrix[state.index,i] = state.emission_vector[i]

    def sample_hypers(self,ialpha,ibeta,igamma):
        k = ibeta.shape[0]-1
        quasi_oracle_matrix = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                if self.transition_matrix[i,j] == 0:
                    quasi_oracle_matrix[i,j] = 0
                else:
                    for l in range(int(self.transition_matrix[i,j])):
                        quasi_oracle_matrix[i,j] += (random.random() < ((self.alpha0 * self.beta[j])/(self.alpha0 * self.beta[j] + l)))

        # print "DEBUG HYPER SAMPLE"
        # print self.beta
        # print self.beta.shape
        # print self.transition_matrix
        # print np.sum(quasi_oracle_matrix, axis=0)
        self.beta = np.asarray(map(lambda x: np.random.gamma(max(x,1),1), np.sum(quasi_oracle_matrix, axis=0)))
        self.beta = np.append(self.beta, np.random.gamma(self.gamma,1))
        # print self.beta
        # print self.beta.shape

        for i in range(self.hyper_resampling_num):
            w = np.random.beta(ialpha + 1, map(lambda x: max(1,x), np.sum(self.transition_matrix, axis=1)))
            p = np.asarray(map(lambda x: max(1,x), np.sum(self.transition_matrix, axis=1)))/ialpha
            p = np.divide(p,p+1)
            s = np.random.binomial(1,p)
            self.alpha0 = np.random.gamma(self.alpha0_a + np.sum(np.sum(quasi_oracle_matrix)) - np.sum(s), (1.0/(self.alpha0_b - np.sum(np.log(w)))))

        k = len(self.beta)
        m = np.sum(np.sum(quasi_oracle_matrix))

        for i in range(self.hyper_resampling_num):
            mu = np.random.beta(igamma + 1, m)
            pi_mu = 1.0 / ((1.0 + (m * (self.gamma_b - np.log(mu)))) / (self.gamma_a + k - 1))
            # print "DEBUG GAMMA INFERENCE"
            # print 1.0/(self.gamma_b - np.log(mu))
            if random.random() < pi_mu:
                self.gamma = np.random.gamma(self.gamma_a + k, 1.0/(self.gamma_b - np.log(mu)))
            else:
                self.gamma = np.random.gamma(self.gamma_a + k - 1, 1.0/(self.gamma_b - np.log(mu)))





    def sample(self,t):
        # print "DEBUG COUNTS"
        # print self.transition_matrix
        # print self.emission_matrix
        # print self.observation_array

        t = t%self.observation_array.shape[0]-1

        if t < self.observation_array.shape[0]:
            ip1 = self.state_array[t+1]
        if t > 0:
            im1 = self.state_array[t-1]
        et = self.observation_array[t]

        self.emission_matrix[self.state_array[t],self.observation_array[t]] -= 1
        if t > 0:
            self.transition_matrix[self.state_array[t],self.state_array[t+1]] -= 1
        if t < self.observation_array.shape[0]:
            self.transition_matrix[self.state_array[t-1],self.state_array[t]] -= 1

        augmented_probabilities = np.zeros(len(self.state_list)+1)

        for _, state in enumerate(self.state_list):

            i = state.index

            if t > 0:

                # print "AUG PROB DEBUG"
                # print i
                # print _
                # print augmented_probabilities[i]
                # print self.transition_matrix[0,i]
                # print self.beta.shape
                # print self.beta[i]

                augmented_probabilities[i] = self.transition_matrix[im1,i] + self.alpha0 * self.beta[i]

            else:

                # print "AUG PROB DEBUG"
                # print i
                # print _
                # print augmented_probabilities[i]
                # print self.transition_matrix[0,i]
                # print self.beta.shape
                # print self.beta[i]

                augmented_probabilities[i] = self.transition_matrix[0,i] + self.alpha0 * self.beta[i]

            if t < self.observation_array.shape[0]:

                if t > 0:

                    if i != self.state_array[t-1]:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:])+self.alpha0))

                    elif i == self.state_array[t-1] and i != self.state_array[t+1]:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:])+self.alpha0 + 1))

                    elif i == self.state_array[t-1] and i == self.state_array[t+1]:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + 1 + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:])+self.alpha0 + 1))

                elif t == 0:
                    if i != 0:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:])+self.alpha0))
                    elif i == 0 and i != self.state_array[t+1]:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:]) + 1 + self.alpha0))
                    elif i == 0 and i == self.state_array[t+1]:
                        augmented_probabilities[i] = augmented_probabilities[i] * ((self.transition_matrix[i,ip1] + 1 + self.alpha0 * self.beta[self.state_array[t+1]])/(np.sum(self.transition_matrix[i,:]) + 1 + self.alpha0))

                augmented_probabilities[i] = augmented_probabilities[i] * ((self.emission_matrix[i,et] + self.emission_priors[et]) / (np.sum(self.emission_matrix[i,:]) + np.sum(self.emission_priors)))

        augmented_probabilities[-1] = ((self.emission_priors[et]/np.sum(self.emission_priors)) * self.alpha0 * self.beta[-1])

        if t < self.observation_array.shape[0]:

            augmented_probabilities[-1] = augmented_probabilities[-1] * self.beta[ip1]

        norm_probabilities = augmented_probabilities / np.sum(augmented_probabilities)

        new_state = 0
        nw_st_tmp = random.random()*float(np.sum(augmented_probabilities))
        for i, tmp in enumerate(augmented_probabilities):
            if np.sum(augmented_probabilities[:i+1]) > nw_st_tmp:
                new_state = i
                break

        if new_state == len(self.state_list):
            self.add_state()

        self.state_array[t] = new_state
        self.state_assignments[t] = self.state_list[new_state]


        self.count_state_emissions()
        self.count_state_transitions()

        i = 0
        while i < len(self.state_list):
            # print "EMPTY STATE MONITOR DEBUG"
            # print len(self.state_list)
            # print self.transition_matrix.shape
            # print i
            if (np.sum(self.transition_matrix[i,:]) + np.sum(self.transition_matrix[:,i])) < 1:
                # print "REMOVED STATE\n\n\n"
                self.remove_state(i)
                i = 0
            i += 1

        self.sample_hypers(self.alpha0,self.beta, self.gamma)

        # print "DEBUG SAMPLE"
        # print "transition matrix"
        # print self.transition_matrix
        # print "emission matrix"
        # print self.emission_matrix
        # print "observation array"
        # print self.observation_array
        # print "state array"
        # print self.state_array
        #
        # print "============================================================="
        #
        # print "aug prob, betas, new state"
        # print augmented_probabilities
        # print self.beta
        # print new_state


class hiddenState:

    def __init__(self, index, parent_model):
        self.index = index
        self.model = parent_model
        self.emission_vector = np.zeros(len(self.model.observation_set))

log = []

input_seq = sys.argv[1]
output_tag = sys.argv[2]

model = hiddenModel(list(input_seq),alpha_a=4.0 ,alpha_b = 2.0, gamma_a = 3, gamma_b = 6)

animation_history_trans = []
animation_history_em = []

model.initialize_random(states=20)
for large in range(100000):
    model.sample(large)
    log.append(len(model.state_list))
    if large%100 == 0:
        animation_history_trans.append(model.transition_matrix)
        animation_history_em.append(model.emission_matrix)

transition_output = open(output_tag + "transitions.txt",mode='w')
emission_output = open(output_tag + "emissions.txt", mode='w')

transition_output.write(str(model.transition_matrix))
emission_output.write(str(model.emission_matrix))

# fig = plt.figure()
# ax = plt.axes()
# im = ax.imshow([])

# def plot_movie(frame):
#     im.set_data(transition_output[frame])
#     return im
#
# anim = FuncAnimation(fig,plot_movie,frames=10000)
# writer = ImageMagickFileWriter()
# anim.save("heatmap_animation.gif", writer=writer fps=100)

plt.figure()
plt.plot(range(100000),log)
plt.savefig(output_tag + "number_of_states.png")

plt.figure()
plt.imshow(self.transition_matrix)
plt.savefig("transition_matrix.png")

plt.figure()
plt.imshow(self.emission_matrix)
plt.savefig("emission_matrix.png")
