import random

RANDOM_SEED = 42  

#Stub function
def rand_eval(feature_set):
    return random.uniform(0, 100)
    
def forward_selection(total_features):
    current_features = [] # Start with no features
    
    initial_accuracy = rand_eval(current_features) # Initial state
    
    # Track best found so far
    best_overall_accuracy = initial_accuracy
    best_overall_features = list(current_features)
    
    print(f"Initial: {initial_accuracy:.1f}%")
    print("Beginning search.\n")
    
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_add_this_level = -1
        
        # Iterate through all features not in set
        for feature in range(1, total_features + 1):
            if feature not in current_features:
                temp_features = current_features + [feature]
                accuracy = rand_eval(temp_features)
                
                print(set(temp_features), accuracy)
                
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = feature
        
        # Keep adding even if accuracy decreases (Search all subsets)
        if feature_to_add_this_level != -1:
            current_features.append(feature_to_add_this_level)
            
            if best_accuracy_this_level > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_features = list(current_features)
                
    print(set(best_overall_features), best_overall_accuracy)

def backward_elimination():
    #TODO: Implement the backward elimination algorithm
    return
    
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

    choice = input("\nEnter choice [1-2]: ").strip()
    if choice == "1":
        forward_selection(total_features)
    elif choice == "2":
        #TODO: Call the backward elimination function
        backward_elimination()
    else:
        print("Invalid Input")

if __name__ == "__main__":
    main()