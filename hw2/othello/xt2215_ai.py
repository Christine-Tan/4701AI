#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
COMS W4701 Artificial Intelligence - Programming Homework 2

An AI player for Othello. This is the template file that you need to  
complete and submit. 

@author: Xinyue Tan xt2215
"""

import random
import sys
import time
from operator import itemgetter;
# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cache = dict();
# 6 is the safely maximum limit for 8*8, 7 risks timeout sometimes.
max_limit = 6;


def compute_utility(board, color):
    """
    Return the utility of the given board state
    (represented as a tuple of tuples) from the perspective
    of the player "color" (1 for dark, 2 for light)
    """
    p1_count, p2_count = get_score(board);
    # utility = n(color) - n(opponent)
    utility = 0;

    if color == 1:
        # 1 for dark
        utility = p1_count - p2_count;
    elif color == 2:
        # 2 for light
        utility = p2_count - p1_count;
    return utility;


############ MINIMAX ###############################

def minimax_min_node(board, color):
    player = 2 if color == 1 else 1;
    successors = get_possible_moves(board, player);
    if len(successors) == 0:
        if board not in cache:
            cache[board] = (compute_utility(board, color), None);
        return cache[board];

    # min the opponent value
    min_v = sys.maxsize;
    min_mov = None;
    for (column, row) in successors:
        new_board = play_move(board, player, column, row);
        # cache
        if new_board not in cache:
            cache[new_board] = minimax_max_node(new_board, color);

        v, mov = cache[new_board];
        if v < min_v:
            min_v = v;
            min_mov = (column, row);

    return min_v, min_mov;


def minimax_max_node(board, color):
    successors = get_possible_moves(board, color);
    if len(successors) == 0:
        if board not in cache:
            cache[board] = (compute_utility(board, color), None);
        return cache[board];

    # max node
    max_v = -sys.maxsize;
    max_mov = None;
    for (column, row) in successors:
        new_board = play_move(board, color, column, row);
        # cache
        if new_board not in cache:
            cache[new_board] = minimax_min_node(new_board, color);

        v, mov = cache[new_board];
        if v > max_v:
            max_v = v;
            max_mov = (column, row);

    return max_v, max_mov;


def select_move_minimax(board, color):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.
    """
    max_v, max_mov = minimax_max_node(board, color);
    return max_mov;


############ ALPHA-BETA PRUNING #####################

def alphabeta_min_node(board, color, alpha, beta, level, limit):
    player = 2 if color == 1 else 1;
    successors = get_possible_moves(board, player);
    if level >= limit == 0 or len(successors) == 0:
        # if (board, level) not in cache:
        #     cache[(board, level)] = (compute_utility(board, color), None);
        # return cache[(board, level)];
        return compute_utility(board, color), None;
    # sort_u : (utility, board)
    sorted_u = list();
    for (column, row) in successors:
        new_board = play_move(board, player, column, row);
        sorted_u.append((compute_utility(new_board, color), new_board, column, row));

    sorted_u.sort(key=itemgetter(0));

    # min the opponent value
    min_v = sys.maxsize;
    min_mov = None;
    # for (column, row) in successors:
    for (u, new_board, column, row) in sorted_u:
        # new_board = play_move(board, player, column, row);

        # cache
        # if (new_board, level + 1) not in cache:
        #     cache[new_board, level + 1] = alphabeta_max_node(new_board, color, alpha, beta, level + 1, limit);
        #
        # v, mov = cache[(new_board, level + 1)];
        if new_board not in cache:
            cache[new_board] = alphabeta_max_node(new_board, color, alpha, beta, level + 1, limit);
        v, mov = cache[new_board];

        if v < min_v:
            min_v = v;
            min_mov = (column, row);
        if v <= alpha:
            return min_v, min_mov;
        beta = min(beta, min_v);
    return min_v, min_mov;


def alphabeta_max_node(board, color, alpha, beta, level, limit):
    successors = get_possible_moves(board, color);
    if level >= limit or len(successors) == 0:
        # if (board, level) not in cache:
        #     cache[(board, level)] = (compute_utility(board, color), None);
        # return cache[(board, level)];
        return compute_utility(board, color), None;

    # sort_u : (utility, board)
    sorted_u = list();
    for (column, row) in successors:
        new_board = play_move(board, color, column, row);
        sorted_u.append((compute_utility(new_board, color), new_board, column, row));

    sorted_u.sort(key=itemgetter(0), reverse=True);

    # max node
    max_v = -sys.maxsize;
    max_mov = None;
    # for (column, row) in successors:
    for (u, new_board, column, row) in sorted_u:
        # new_board = play_move(board, color, column, row);

        # cache
        # if (new_board, level + 1) not in cache:
        #     cache[(new_board, level + 1)] = alphabeta_min_node(new_board, color, alpha, beta, level + 1, limit);
        # v, mov = cache[(new_board, level + 1)];

        if new_board not in cache:
            cache[new_board] = alphabeta_min_node(new_board, color, alpha, beta, level + 1, limit);
        v, mov = cache[new_board];

        if v > max_v:
            max_v = v;
            max_mov = (column, row);
        if v >= beta:
            return max_v, max_mov;

        alpha = max(alpha, max_v);
    return max_v, max_mov;


def select_move_alphabeta(board, color):
    alpha = -sys.maxsize;
    beta = sys.maxsize;
    max_v, max_mov = alphabeta_max_node(board, color, alpha, beta, 0, max_limit);
    return max_mov;


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Minimax AI")  # First line is the name of this AI
    color = int(input())  # Then we read the color: 1 for dark (goes first),
    # 2 for light.

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL":  # Game is over.
            print
        else:
            board = eval(input())  # Read in the input and turn it into a Python
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            # Select the move and send it to the manager
            # movei, movej = select_move_minimax(board, color)
            movei, movej = select_move_alphabeta(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
