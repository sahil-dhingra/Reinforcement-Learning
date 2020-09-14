# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:06:41 2019

@author: sahil.d
"""

import random

class soccer:
    # Board
    board = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1],
             [2, 0],
             [2, 1],
             [3, 0],
             [3, 1]]
    
    # Eligible moves
    moves = [[0,-1],
             [-1,0],
             [0, 0],
             [1, 0],
             [0, 1]]
    
    # Goal positions
    goal_A =[[0, 0],
             [0, 1]]
    goal_B =[[3, 0],
             [3, 1]]
        
    def __init__(self, width=4, height=2):
        # Player & Ball initial positions
        self.player_A = [2, 1]
        self.player_B = [1, 1]
        self.position_ball = [1, 1]
        self.ball_with = 'B'
        self.position_players = {'A' : self.player_A, 'B' : self.player_A}
        self.is_over = None
        self.goal = None
        self.order = None
        
        # Random start
        self.ball_with = ['A','B'][random.randrange(2)]
        p_ball = 2+random.randrange(4)
        self.position_ball = self.board[p_ball]
        free_spots = list(range(2,p_ball)) + list(range(p_ball+1,6))
        self.player_A = self.position_ball if self.ball_with == 'A' else self.board[random.choice(free_spots)]
        self.player_B = self.position_ball if self.ball_with == 'B' else self.board[random.choice(free_spots)]
    
    # Simulate actions for A & B
    def play(self, action_A_i, action_B_i):
        action_A = self.moves[action_A_i]
        action_B = self.moves[action_B_i]
        if random.randrange(2) == 0:
            self.order = ['A','B']
            for i in range(1):
                self.move_player(self.player_A, self.player_B, action_A, 'A')
                if self.goal:
                    self.is_over = 1
                    break
                self.move_player(self.player_B, self.player_A, action_B, 'B')
                if self.goal:
                    self.is_over = 1
                    break
        else:
            self.order = ['B','A']
            for i in range(1):
                self.move_player(self.player_B, self.player_A, action_B, 'B')
                if self.goal:
                    self.is_over = 1
                    break
                self.move_player(self.player_A, self.player_B, action_A, 'A')  
                if self.goal:
                    self.is_over = 1
                    break
    
    # Execute actions and modify environment states
    def move_player(self, player, opp_player, action, player_tag):
        new_player = list(map(sum, zip(player, action)))
        valid_move = 0
        collision = 0
        
        # Collision check
        if new_player == opp_player:
            collision = 1
            self.position_ball = opp_player
            if self.ball_with == 'B' and player_tag == 'B':
                self.ball_with = 'A'
            if self.ball_with == 'A' and player_tag == 'A':
                self.ball_with = 'B'
            
        # Bounds check
        if new_player in self.board:
            valid_move = 1
        
        # Goal conditions
        if new_player in self.goal_A and self.ball_with == player_tag:
            self.goal = 'A'
        if new_player in self.goal_B and self.ball_with == player_tag:
            self.goal = 'B'

        # Updating player positions
        if player_tag == 'A' and valid_move == 1 and collision == 0:
            self.player_A = new_player
            self.position_ball = new_player if self.ball_with == 'A' else self.position_ball
        if player_tag == 'B' and valid_move == 1 and collision == 0:
            self.player_B = new_player
            self.position_ball = new_player if self.ball_with == 'B' else self.position_ball                    
    
    # Return current state of the game in int form
    def i_state(self):
        A = 2*self.player_A[0] + self.player_A[1]
        B = 2*self.player_B[0] + self.player_B[1]
        Ball = 0 if self.ball_with == 'A' else 1
        return (A, B, Ball)
    
    # Return actions in int form
    def i_actions(self, action_A, action_B):
        aA = action_A[0] + 2*action_A[1] + 2
        aB = action_B[0] + 2*action_B[1] + 2
        return [aA, aB] 
    
    # Return rewards for [A, B]
    def get_reward(self):
        reward = [0, 0]
        if self.goal:
            if self.goal == 'A':
                reward = [100, -100]
            if self.goal == 'B':
                reward = [-100, 100]
        return reward
    
    # Game state
    def game_state(self):
        return self.player_A, self.player_B, self.position_ball, self.is_over, self.ball_with, self.goal
    
    # Reset game
    def reset(self):
        self.__init__()
