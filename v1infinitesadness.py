#!/usr/bin/env python
from __future__ import division
import sys
import numpy as np
import random
import copy
from scipy.special import expit



def pick_from_odds(vector):
    if np.nan in vector:
        raise Exception("Invalid Odds")
    if np.inf in vector:
        return list(vector).index(np.inf)


    probability = np.divide(vector,(1.+vector))

    print "Probability vector debug"
    print vector
    print probability
    print np.sum(probability)

    temp = 0.
    pick = random.random()
    for i, element in enumerate(vector):
        temp += element
        if temp > pick:
            return i
    # print "RAN OFF THE END"
#    raw_input()
    return -1



def odds(vector,log=False, x = 0, alpha = 0, beta = 0, gamma = 0, ai = 0, oracle_vector=None, aux_state=False):
    if oracle_vector == None:
        oracle_vector = np.ones(vector.shape,dtype=float)
    oracle_probabilities = np.divide(oracle_vector,(np.sum(oracle_vector)+gamma)) * (float(beta)/float(np.sum(vector)+beta))
    # print "ODDS DEBUG"
    # print vector
    # print beta
    # print oracle_probabilities
    vector[ai] += alpha
    if aux_state:
        gamma_prob = (float(beta)/float(np.sum(vector) + x + alpha + beta)) * (float(float(gamma)/ float(np.sum(oracle_vector)+gamma)))
    probabilities = (vector.astype(dtype=float)/float(np.sum(vector) +x + alpha + beta)) + oracle_probabilities
    if aux_state:
        probabilities = np.append(probabilities,np.array([gamma_prob]))
    # print "TOTAL PROBABILITIES"
    # print probabilities
    # print "PROBABILITY SUM"
    # print np.sum(probabilities)
    odds = np.divide(probabilities,np.ones(probabilities.shape,dtype=float)-probabilities)
    # print odds
    if log == True:
         log_odds = np.log(odds)
        #  print log_odds
         #raw_input()
         return log_odds
    else:
        return odds

class hiddenModel:

    def __init__(self,observations, alpha=1,beta=1,gamma=10,betaE=1,gammaE=100):

        self.observations = observations
        self.observation_array = np.asarray(observations)
        self.observation_set = set(self.observations)
        self.observation_index_list = list(self.observation_set)

        self.state_assignments = []
        self.state_list = []
        self.transition_dict = {}

        self.transition_matrix = np.zeros((0,0))
        self.emission_matrix = np.zeros((0,len(self.observation_set)))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.betaE = betaE
        self.gammaE = gammaE

        self.state_oracle = infiniteStateOracle(self)

    # def generative_process(self):
    #     self.add_state()
    #     self.state_assignments.append(self.state_list[0])
    #     for i, observation in enumerate(self.observations[1:]):
    #         self.state_assignments[i-1]

    def update_hypers(self):
        new_beta = np.sum(self.state_oracle.oracle_vector)
        new_gamma = float(np.sum(self.state_oracle.oracle_vector))/float(len(self.state_list))
        new_betaE = np.sum(self.state_oracle.emission_oracle_matrix)
        new_gammaE = float(np.sum(self.state_oracle.emission_oracle_matrix))/float(self.state_oracle.emission_oracle_matrix.shape[1])
        print "New Hypers"
        print new_beta
        print new_gamma
        print new_betaE
        print new_gammaE
        raw_input()

        self.beta = new_beta
        self.gamma = new_gamma
        self.betaE = new_betaE
        self.gammaE = new_gammaE
        self.state_oracle.beta = new_beta
        self.state_oracle.gamma = new_gamma
        self.state_oracle.gammaE = new_gammaE
        self.state_oracle.betaE = new_betaE
        for emission_oracle in self.state_oracle.emission_oracle_list:
            emission_oracle.betaE = new_betaE
            emission_oracle.gammaE = new_gammaE


    def initialize_random(self,states=10):
        for i in range(states):
            self.add_state()
        for observation in self.observations:
            self.state_assignments.append(random.choice(self.state_list))
        self.count_state_transitions()
        self.state_oracle.count_oracle_vector()
        for state in self.state_list:
            state.count_emissions()
        print self.transition_matrix



    def add_state(self):
        self.state_list.append(hiddenState(len(self.state_list),self))
        for state1 in self.state_list:
            for state2 in self.state_list:
                if (state1,state2) not in self.transition_dict:
                    self.transition_dict[(state1,state2)] = []
        self.transition_matrix = np.append(self.transition_matrix,np.zeros((1,self.transition_matrix.shape[1])),axis=0)
        self.transition_matrix = np.append(self.transition_matrix,np.zeros((self.transition_matrix.shape[0],1)),axis=1)
        self.emission_matrix = np.append(self.emission_matrix,np.zeros((1,self.emission_matrix.shape[1])),axis=0)
        self.state_oracle.add_state()

    def remove_state(self,state):
        self.state_list.pop(state.index)
        for i, item in enumerate(self.state_list):
            item.index = i
        self.transition_matrix = np.delete(self.transition_matrix, state.index, axis=0)
        self.transition_matrix = np.delete(self.transition_matrix, state.index, axis=1)
        self.emission_matrix = np.delete(self.emission_matrix, state.index, axis=0)
        self.count_state_transitions()
        self.state_oracle.remove_state(state.index)

    def count_state_transitions(self):
        #temp_transition_dict = np.zeros((len(self.state_list),len(self.state_list)))
        self.transition_matrix = np.zeros(self.transition_matrix.shape)
        for i, state in enumerate(self.state_assignments[:-1]):
            #temp_transition_dict[state_list.index(state),state_list.index(self.state_assignments[i+1])] += 1
            self.transition_matrix[self.state_assignments[i].index,self.state_assignments[i+1].index] += 1
        print self.transition_matrix
        #return temp_transition_dict


    def sample(self,passed_i):

        index = passed_i%len(self.observations)
        if passed_i%len(self.observations) == len(self.observations)-1:
            self.update_hypers()


        #     if random.random() < .1:

        #         self.state_oracle.reset()
        #         raw_input()
#        index = random.randint(0,len(self.observations)-1)
        print "Observation index"
        print index

        emission_index = self.observation_index_list.index(self.observations[index])

        temp_transition_matrix = copy.deepcopy(self.transition_matrix)
        temp_emission_matrix = copy.deepcopy(self.emission_matrix)
        print "Actual observation at index"
        print self.observations[index]

        print self.state_assignments[index]

        old_state = self.state_assignments[index]
        print "St state index (in matricies)"
        print old_state.index



        # temp_transition_matrix[old_state.index,self.state_assignments[index+1].index] -= 1
        # temp_transition_matrix[self.state_assignments[index-1].index,old_state.index] -= 1
        temp_emission_matrix[self.state_assignments[index].index,self.observation_index_list.index(self.observations[index])] -= 1
        # #
        # print "s+1"
        # print self.state_assignments[index+1].index
        # print "s-1"
        # print self.state_assignments[index-1].index
        #
        # print "New values block"
        # print temp_transition_matrix[old_state.index]
        # print temp_transition_matrix[:,old_state.index]
        # print temp_emission_matrix[old_state.index]
        # print temp_transition_matrix
        # print "End new values block"


        # if np.sum(temp_transition_matrix[old_state.index]) == 0:
        #     self.remove_state(old_state)
        #     print "Empty state removed"
        #     raw_input()
        # if np.sum(temp_transition_matrix[:,old_state.index]) == 0:
        #     self.remove_state(old_state)
        #     print "Empty empty removed"
        #     raw_input()

        print "Emission matrix, raw"
        print temp_emission_matrix
        print "transition_matrix_raw"
        print temp_transition_matrix

            # if np.sum(temp_transition_matrix[old_state.index]) == 0:
            #     print "Empty emission with extant transitions"
            #     print temp_transition_matrix[old_state.index]
            #     print temp_emission_matrix[old_state.index]
            #     raw_input()

        if index > 0:
            priors_s1_counts = temp_transition_matrix[self.state_assignments[index-1].index,:]
            print "State index of St-1"
            print self.state_assignments[index-1].index
        else:
            priors_s1_counts = np.ones(self.emission_matrix.shape[0])
        if index < (len(self.observations)-1):
            priors_s3_counts = temp_transition_matrix[:,self.state_assignments[index+1].index]
            print "State index of St+1"
            print self.state_assignments[index+1].index
        else:
            priors_s3_counts = np.ones(priors_s1_counts.shape)



        # if np.sum(priors_s1_counts) == 0:
        #     print old_state.index
        #     raw_input()
        # if np.sum(priors_s3_counts) == 0:
            # print old_state.index
            # raw_input()
        #if np.sum(emission_priors_counts) == 0:
        #    raw_input()
        #    print old_state.index


        alter_priors_s1 = odds(priors_s1_counts, log=True, beta = self.beta, gamma=self.gamma, oracle_vector = self.state_oracle.oracle_vector)
        # priors_s3 = odds(priors_s3_counts, log=True, beta = self.beta, gamma=self.gamma, oracle_vector = self.state_oracle.oracle_vector)

        lr_s1 = np.ones(temp_transition_matrix.shape[0]+1,dtype=float)


        if index > 0:

            s1_state_index = self.state_assignments[index-1].index


            for i, state in enumerate(temp_transition_matrix):


                emissions_of_state_given_previous = state[s1_state_index] + (float(self.beta) * (float(self.state_oracle.oracle_vector[i])/float(np.sum(self.state_oracle.oracle_vector)+self.gamma)))


                total_emissios_of_previous_state = np.sum(temp_transition_matrix[s1_state_index]) + self.beta

                total_emissions_of_state_given_NOT_previous = np.sum(temp_transition_matrix[:,i]) - temp_transition_matrix[s1_state_index,i] + (self.beta * (np.sum(self.state_oracle.oracle_vector)-self.state_oracle.oracle_vector[i]))

                total_emissions_given_NOT_previous = np.sum(temp_transition_matrix) - np.sum(temp_transition_matrix[s1_state_index]) + (self.beta * np.sum(self.state_oracle.oracle_vector))

                print "Debug sequence priors"
                print emissions_of_state_given_previous
                print total_emissios_of_previous_state
                print total_emissions_of_state_given_NOT_previous
                print total_emissions_given_NOT_previous

                lr_s1[i] = (float(emissions_of_state_given_previous)/float(total_emissios_of_previous_state))/(float(total_emissions_of_state_given_NOT_previous)/float(total_emissions_given_NOT_previous))

                print lr_s1[i]

            s1_trans_to_gamma = (self.beta / (np.sum(temp_transition_matrix) + self.beta)) * (self.gamma/(np.sum(self.state_oracle.emission_oracle_matrix[s1_state_index])+self.gamma))

            NOT_s1_trans_to_gamma = ((self.beta * (len(self.state_list)-1)) / (np.sum(temp_transition_matrix)-np.sum(temp_emission_matrix[s1_state_index])+(self.beta * len(self.state_list)))) * ((self.gamma*(len(self.state_list)-1))/ (np.sum(self.state_oracle.state_oracle_emissions)-self.state_oracle.state_oracle_emissions[s1_state_index]+ (self.gamma*(len(self.state_list)-1))))

            print "DEBUG NEW STATE"


            print s1_trans_to_gamma
            print NOT_s1_trans_to_gamma

            lr_s1[-1] = s1_trans_to_gamma/NOT_s1_trans_to_gamma

            print lr_s1
            print expit(lr_s1)
            print sum(expit(lr_s1))


        # if index < (len(self.observations)-1):
        #
        #     s3_state_index = self.state_assignments[index+1].index
        #
        #     lr_s3 = np.ones(temp_transition_matrix.shape[0]+1,dtype=float)
        #
        #     for i, state in enumerate(temp_transition_matrix):
        #
        #
        #         emission_of_next_given_state = state[s3_state_index] + (float(self.beta) * (float(sum(self.state_oracle.emission_oracle_matrix[old_state.index,s3_state_index]))/float(np.sum(self.state_oracle.emission_oracle_matrix[s3_state_index]))))
        #
        #         total_emissios_of_next_state = np.sum(temp_transition_matrix[:,s3_state_index]) + self.beta
        #
        #         total_emissions_of_next_given_NOT_state = np.sum(temp_transition_matrix[:,s3_state_index]) - temp_transition_matrix[old_state.index,s3_state_index] + (self.beta * ((np.sum(self.state_oracle.emission_oracle_matrix[:,s3_state_index])-self.state_oracle.emission_oracle_matrix[old_state.index,s3_state_index]))/(np.sum(self.state_oracle.emission_oracle_matrix)-sum(self.state_oracle.emission_oracle_matrix[old_state.index,:])+self.gamma))
        #
        #         total_emissions_given_NOT_state = np.sum(temp_transition_matrix) - np.sum(temp_transition_matrix[old_state.index,:]) + (self.beta * len(self.state_list))
        #
        #         lr_s3[i] = (float(emissions_of_state_given_previous)/float(total_emissios_of_previous_state))/(float(total_emissions_of_state_given_NOT_previous)/float(total_emissions_given_NOT_previous))
        #
        #     gamma_to_s3 = 1./
        #
        #     s3_from_NOT_gamma = ((self.beta * (len(self.state_list)-1)) / (np.sum(temp_transition_matrix)-np.sum(temp_emission_matrix[s1_state_index])+(self.beta * len(self.state_list)))) * ((self.gamma*(len(self.state_list)-1))/ (np.sum(self.state_oracle.emission_oracle_matrix)-self.state_oracle.emission_oracle_matrix[s1_state_index]+ (self.gamma*(len(self.state_list)-1)))
        #
        #     lr_s3[-1] = s1_trans_to_gamma/NOT_s1_trans_to_gamma




        emission_lr = np.ones(temp_transition_matrix.shape[0],dtype=float)

        for i, state_emissions in enumerate(self.emission_matrix):

            expected_emission_observed_given_state = float(state_emissions[emission_index]) + float(self.betaE * float(self.state_oracle.emission_oracle_list[i].emission_oracle_vector[emission_index] / float(np.sum(self.state_oracle.emission_oracle_list[i].emission_oracle_vector))))

            total_emissions_expected_given_state = float(np.sum(self.emission_matrix[i])+self.betaE)


            expected_emission_given_NOT_state = np.sum(self.emission_matrix[:,emission_index]) - state_emissions[emission_index] + self.betaE*(float(sum(map(lambda x: x.emission_oracle_vector[emission_index],self.state_oracle.emission_oracle_list)) -self.state_oracle.emission_oracle_list[i].emission_oracle_vector[emission_index])/float(sum(map(lambda x: np.sum(x.emission_oracle_vector),self.state_oracle.emission_oracle_list))-np.sum(self.state_oracle.emission_oracle_list[i].emission_oracle_vector)+self.gammaE))


            expected_total_emissions_given_NOT_state = sum(map(lambda x: np.sum(x.emission_oracle_vector),self.state_oracle.emission_oracle_list)) - np.sum(self.state_oracle.emission_oracle_list[i].emission_oracle_vector) + (float(self.betaE*(self.transition_matrix.shape[0]-1)))

            emission_lr[i] = (float(expected_emission_observed_given_state)/float(total_emissions_expected_given_state))/(float(expected_emission_given_NOT_state)/float(expected_total_emissions_given_NOT_state))

            print "Debug emission priors"
            print expected_emission_observed_given_state
            print total_emissions_expected_given_state
            print expected_emission_given_NOT_state
            print expected_total_emissions_given_NOT_state
            print emission_lr[i]

        e_prob_given_gamma = 1./len(self.observation_set)

        e_prob_given_not_gamma_numerator = self.beta*(float(self.gamma*len(self.state_list))/float(sum(map(lambda x: np.sum(x.emission_oracle_vector),self.state_oracle.emission_oracle_list)) + self.gamma))

        e_prob_given_not_gamma_denominator = sum(map(lambda x: np.sum(x.emission_oracle_vector),self.state_oracle.emission_oracle_list)) + (float(self.betaE*(self.transition_matrix.shape[0])))

        gamma_lr = float(e_prob_given_gamma) / (float(e_prob_given_not_gamma_numerator)/float(e_prob_given_not_gamma_denominator))

        emission_lr = np.append(emission_lr, np.array([gamma_lr]))



#         emission_likelihood_ratio_matrix = np.zeros(priors_s1.shape)
#         for i, state in enumerate(self.emission_matrix):
#             # print "Emissions of observation from state"
#             # print float(state[emission_index])
#             # print "Total emissions from state, plus beta"
#             # print float(np.sum(state)+self.beta)
#             # print "Emissions of observation from other states"
#             emission_likelihood_ratio_matrix[i] = (float(state[emission_index]+(float(self.betaE)*(float(self.state_oracle.oracle_vector[i])/float(np.sum(self.state_oracle.oracle_vector))))/float(np.sum(state)+self.betaE))/ \
# \
#              (float(np.sum(self.emission_matrix[:,emission_index])-state[emission_index] + self.betaE * (float(self.state_oracle.oracle_vector[i])/float(np.sum(self.state_oracle.oracle_vector))))/ float(np.sum(self.emission_matrix)-np.sum(state)+(self.betaE*self.emission_matrix.shape[0]-1))))

        print "GAMMA EXCLUSION"

#        priors = odds(np.ones(temp_transition_matrix.shape[0]),beta = 1, gamma = 1, aux_state=True, log=True)

        priors = np.zeros(emission_lr.shape)


        log_r_s1 = np.log(lr_s1)

        emission_log_odds = np.log(emission_lr)

        print "Log odds likelihood of each state according to St-1, St+1, and the observed emission"
        # print priors_s1
        # print priors_s3
        print priors
        print log_r_s1
        print emission_log_odds
        print "###########################"


#        log_odds_posteriors = (priors_s1 + priors_s3)/2. + emission_log_odds

        log_odds_posteriors = priors + log_r_s1 + 2*emission_log_odds

        posterior_odds = np.exp(log_odds_posteriors)

        print "Posterior odds of each state"
        print posterior_odds

        new_pick = pick_from_odds(posterior_odds)
        print "New state?"
        print new_pick
        print len(posterior_odds)

        if new_pick != len(posterior_odds)-1:
            new_state = self.state_list[new_pick]

            if index > 0:
                prob_normal_transition = float(temp_transition_matrix[self.state_assignments[index-1].index,new_state.index]) / float((np.sum(temp_transition_matrix[self.state_assignments[index-1].index,:]) + self.beta))
                prob_beta_transition = (float(self.beta) / float((np.sum(temp_transition_matrix[self.state_assignments[index-1].index,:]) + self.beta))) * (float(self.state_oracle.oracle_vector[self.state_assignments[index-1].index])/float(np.sum(self.state_oracle.oracle_vector)+self.gamma))

                print "Oracle Transition?"
                print prob_normal_transition
                print prob_beta_transition

                if random.random() < prob_beta_transition/(prob_normal_transition+prob_beta_transition):
                    self.state_oracle.state_oracle_emissions[index-1] = 1
                else:
                    self.state_oracle.state_oracle_emissions[index-1] = 0

            # if index < (len(self.observations)-1):
            #     prob_normal_transition = float(temp_transition_matrix[new_state.index,self.state_assignments[index+1].index]) / float((np.sum(temp_transition_matrix[:,self.state_assignments[index+1].index]) + self.beta))
            #     prob_beta_transition = (float(self.beta) / float((np.sum(temp_transition_matrix[:,self.state_assignments[index+1].index]) + self.beta))) * (float(self.state_oracle.oracle_vector[self.state_assignments[index+1].index])/float(np.sum(self.state_oracle.oracle_vector)+self.gamma))
            #     if random.random() < prob_beta_transition/(prob_normal_transition+prob_beta_transition):
            #         self.state_oracle.state_oracle_emissions[index+1] = 1
            #     else:
            #         self.state_oracle.state_oracle_emissions[index-1] = 0


            prob_normal_emission = float(temp_emission_matrix[new_state.index,emission_index]) / float(np.sum(temp_emission_matrix[new_state.index,:])+self.betaE)
            prob_beta_emission = (float(self.betaE) / float(np.sum(temp_emission_matrix[new_state.index,:])+self.betaE)) * (self.state_oracle.emission_oracle_list[new_state.index].emission_oracle_vector[emission_index]) / float(np.sum(self.state_oracle.emission_oracle_list[new_state.index].emission_oracle_vector)+self.gammaE)

            print "Oracle Emission?"
            print prob_normal_emission
            print prob_beta_emission

            if random.random() < prob_normal_emission / (prob_normal_emission + prob_beta_emission):
                self.state_oracle.emission_oracle_emissions[index] = 1
            else:
                self.state_oracle.emission_oracle_emissions[index] = 0

            print "Transition oracle"
            print self.state_oracle.oracle_vector
            print self.state_oracle.emission_oracle_matrix

        else:
            print "Add"
            self.add_state()
            new_state = self.state_list[-1]

        print "New state chosen"
        print new_state.index

        print "New state emission frequencies"
        print self.emission_matrix[new_state.index]

        self.state_assignments[index] = new_state


        # if passed_i%len(self.observations) == 0:
        self.count_state_transitions()
        self.state_oracle.count_oracle_vector()
        for state in self.state_list:
            state.count_emissions()
            if np.sum(self.emission_matrix[state.index]) < 1:
                self.remove_state(state)
                print "Delete"
                #raw_input()



        # temp_transition_matrix[new_state.index,self.state_assignments[index+1].index] += 1
        # temp_transition_matrix[self.state_assignments[index-1].index,new_state.index] += 1
        # temp_emission_matrix[new_state.index,self.observation_index_list.index(self.observations[index])] += 1
        #
        #
        #
        #
        # self.transition_matrix = temp_transition_matrix
        # self.emission_matrix = temp_emission_matrix




    def forward_algorithm():
        pass



class infiniteStateOracle:




    def __init__(self, parent_model):
        self.parent_model = parent_model
        self.oracle_vector = np.zeros(0)
        self.emission_oracle_matrix = np.zeros((0,len(parent_model.observation_set)))
        self.alpha = parent_model.alpha
        self.beta = parent_model.beta
        self.gamma = parent_model.gamma
        self.betaE = parent_model.betaE
        self.gammaE = parent_model.gammaE

        self.emission_oracle_list = []
        self.state_oracle_emissions = (np.random.random(len(parent_model.observations)) < .2).astype(dtype=int)
        self.emission_oracle_emissions = (np.random.random(len(parent_model.observations)) < .2).astype(dtype=int)
        self.state_assignment_array = np.asarray(map(lambda x: x.index,parent_model.state_assignments))

    def reset(self):
        self.state_oracle_emissions = np.ones(self.state_oracle_emissions.shape)
        for emission_oracle in self.emission_oracle_list:
            emission_oracle.reset()
        self.count_oracle_vector()

    def count_oracle_vector(self):
        self.state_assignment_array = np.asarray(map(lambda x: x.index,self.parent_model.state_assignments))

        for i,state in enumerate(self.oracle_vector):
            state = np.sum(self.state_oracle_emissions[self.state_assignment_array == i])
            if state < len(self.parent_model.observation_set):
                state = len(self.parent_model.observation_set)
        for i, emission_oracle in enumerate(self.emission_oracle_list):
            filter_by_state = self.state_assignment_array == i
            for j, observation in enumerate(self.parent_model.observation_index_list):
                filter_by_observation = self.parent_model.observation_array == observation
                total_filter = np.logical_and(filter_by_state,filter_by_observation)
                emission_oracle.emission_oracle_vector[self.parent_model.observation_index_list.index(observation)] = max(np.sum(self.emission_oracle_emissions[total_filter]),1)
                self.emission_oracle_matrix[i,j] = max(np.sum(self.emission_oracle_emissions[total_filter]),1)

    def add_state(self):
        self.oracle_vector = np.append(self.oracle_vector, np.ones(1))
        self.emission_oracle_list.append(infiniteEmissionOracle(self,len(self.parent_model.observation_set)))
        print "Oracle Matrix Shape"
        print self.emission_oracle_matrix.shape
        self.emission_oracle_matrix = np.append(self.emission_oracle_matrix,np.ones((1,len(self.parent_model.observation_set))),axis=0)



    def remove_state(self,i):
        self.oracle_vector = np.delete(self.oracle_vector,i,axis=0)
        self.emission_oracle_list.pop(i)
        self.emission_oracle_matrix = np.delete(self.emission_oracle_matrix,i,axis=0)


    def sample(self, i):
        temp_transition_matrix = self.model.transition_matrix
        temp_emission_matrix = self.model.emission_matrix
        old_state = self.model.state_assignments[i]



    def choose_state(self):
        span = np.sum(self.oracle_vector) + self.gamma
        choice = random.random()*float(span)
        temp = 0
        for i, state in enumerate(self.oracle_vector):
            temp += state
            if temp > choice:
                self.oracle_vector[i] += 1
                return(i)
            else:
                return -1

class infiniteEmissionOracle:



    def __init__(self, parent_state_oracle, vocabulary):
        pass
        self.parent_oracle = parent_state_oracle
        self.betaE = self.parent_oracle.betaE
        self.gammaE = self.parent_oracle.gammaE
        self.emission_oracle_vector = np.ones(vocabulary)

    def reset(self):
        self.emission_oracle_vector = np.ones(self.emission_oracle_vector.shape)

    def sample_emission(self):
        pass

class hiddenState:

    def __init__(self, index, parent_model):
        self.emission_counts = {}
        self.index = index
        self.model = parent_model

    def count_emissions(self):
        observation_list = []
        for i, observation in enumerate(self.model.observations):
            if self.model.state_assignments[i].index == self.index:
                observation_list.append(self.model.observations[i])
        print observation_list
        for observation in self.model.observation_set:
            self.emission_counts[observation] = 0
        for observation in observation_list:
            self.emission_counts[observation] += 1
        print self.emission_counts

        self.emission_vector = np.zeros(len(self.emission_counts))
        for key in self.emission_counts:
            self.emission_vector[self.model.observation_index_list.index(key)] = self.emission_counts[key]
        self.model.emission_matrix[self.index] = self.emission_vector
        print self.emission_vector


observation_concat = list("ABCDCB")*20


model = hiddenModel(observation_concat,beta=1,betaE=.01,gamma=30,gammaE=10)

model.initialize_random(states=20)

print observation_concat

print model.state_list
print map(lambda x: x.index, model.state_assignments)
print len(model.state_assignments)
print len(model.observations)
print model.emission_matrix
print model.transition_matrix

for i in range(10000):
    print i
    model.sample(i)
    if i%500 == 0:
        print map(lambda x: x.index, model.state_assignments)
        raw_input()

for state in model.state_assignments:
    print model.observation_index_list[pick_from_odds(odds(model.emission_matrix[state.index]))]
