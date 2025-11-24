import random
import itertools
from typing import Callable, List, Set, Tuple

RANDOM_SEED = 42  

def rand_eval():
    #TODO: Implement a random evaluation function
    
def forward_selection():
    #TODO: Implement the forward selection algorithm

def backward_elimination():
    #TODO: Implement the backward elimination algorithm
    
def main():

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("Welcome to Eric, Rafat, and Lynvy's Feature Selection Algorithm.")

    while True:
        try:
            total_features = int(input("\nPlease enter total number of features: ").strip())
            if total_features <= 0:
                raise ValueError    
            break
        except ValueError:
            print("Please enter a positive integer for number of features.")

    print("\nType the number of the algorithm you want to run.\n")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input("\nEnter choice [1-3]: ").strip()
    if choice == "1":
        #TODO: Call the forward selection function
        forward_selection()
    elif choice == "2":
        #TODO: Call the backward elimination function
        backward_elimination()
    else:
        print("Invalid Input")



