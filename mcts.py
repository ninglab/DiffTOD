import math
import random
import numpy as np

class Node():
    def __init__(self):
        ''' Initialize a new state '''
        self.num_visited = 0
        self.total_reward = 0
        self.r = 0#immediate reward
        self.Q = 0
        self.action = None
        self.parent = None
        self.children = {}
        self.is_root = False
        self.is_leaf = False
        self.dialogue_history_so_far = ""

    def update(self,reward):
        ''' Update Q Value of this Node'''
        self.num_visited += 1
        self.total_reward += reward
        self.Q = self.total_reward/self.num_visited

    def calculate_policy(self):
        temp = sum([child.Q for child in children.values()])
        policy_dict = {}
        for key, value in self.children.items():
            policy_dict[key] = value.Q/temp
        return policy_dict

    def select(self, w):
        max_UCT = -float("inf")
        selected_action = None
        selected_children = None
        for key, value in self.children.items():
            if value is None:
                UCT = w * (np.sqrt(np.log(self.num_visited+1)))
            else:
                UCT = value.Q + w * (np.sqrt(np.log(self.num_visited+1))/(value.num_visited+1))
            if UCT > max_UCT:
                selected_action = key
                selected_children = value
                max_UCT = UCT
        return selected_action, selected_children
    
    def expand(self, action_to_expand):
        # rememeber to update the dialogue_history_so_far for the new node
        new_node = Node()
        new_node.action = action_to_expand
        new_node.parent = self
        new_node.children = dict.fromkeys(self.children.keys(), None)
        new_node.is_leaf = True
        self.children[action_to_expand] = new_node
        return new_node
    
    def back_propagate(self, cumulative_reward, discount_factor):
        #cumulative reward up until this turn (including this turn)
        self.update(cumulative_reward)
        if self.parent != None:
            cumulative_reward = self.parent.r + discount_factor * cumulative_reward
            self.parent.back_propagate(cumulative_reward, discount_factor)