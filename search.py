import random

RANDOM_SEED = 42  

def rand_eval(feature_set):
    return random.uniform(0, 100)
    
def forward_selection(total_features):
    current_features = [] # Start with no features
    
    initial_accuracy = rand_eval(current_features) # Initial state
    
    # Track best found so far
    best_overall_accuracy = initial_accuracy
    best_overall_features = list(current_features)
    
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {initial_accuracy:.1f}%")
    print("Beginning search.\n")
    
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_add_this_level = -1
        
        # Iterate through all features not in set
        for feature in range(1, total_features + 1):
            if feature not in current_features:
                temp_features = current_features + [feature]
                accuracy = rand_eval(temp_features)
                
                print(f"\tUsing feature(s) {set(temp_features)} accuracy is {accuracy:.1f}%")
                
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = feature
        
        # Keep adding even if accuracy decreases (Search all subsets)
        if feature_to_add_this_level != -1:
            current_features.append(feature_to_add_this_level)
            print(f"\nFeature set {set(current_features)} was best, accuracy is {best_accuracy_this_level:.1f}%")
            
            if best_accuracy_this_level > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_features = list(current_features)
            else:
                print("(Warning, Accuracy has decreased!)")
                
    print(f"\nFinished search!! The best feature subset is {set(best_overall_features)}, which has an accuracy of {best_overall_accuracy:.1f}%")

def backward_elimination(total_features):
    current_features = list(range(1, total_features + 1)) # Start with all features
    
    initial_accuracy = rand_eval(current_features) # Initial state
    
    # Track best found so far
    best_overall_accuracy = initial_accuracy
    best_overall_features = list(current_features)
    
    print(f"Using all features {set(current_features)} and \"random\" evaluation, I get an accuracy of {initial_accuracy:.1f}%")
    print("Beginning search.\n")
    
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_remove_this_level = -1
        
        # Iterate through all features in set to find best removal
        for feature in current_features:
            temp_features = [f for f in current_features if f != feature]
            accuracy = rand_eval(temp_features)
            
            print(f"\tUsing feature(s) {set(temp_features)} accuracy is {accuracy:.1f}%")
            
            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                feature_to_remove_this_level = feature
                
        if feature_to_remove_this_level != -1:
            current_features.remove(feature_to_remove_this_level)
            print(f"\nFeature set {set(current_features)} was best, accuracy is {best_accuracy_this_level:.1f}%")
            
            if best_accuracy_this_level >= best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_features = list(current_features)
            else:
                print("(Warning, Accuracy has decreased!)")

    print(f"\nFinished search!! The best feature subset is {set(best_overall_features)}, which has an accuracy of {best_overall_accuracy:.1f}%")

    
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
        #TODO: Call the forward selection function
        forward_selection(total_features)
    elif choice == "2":
        #TODO: Call the backward elimination function
        backward_elimination(total_features)
    else:
        print("Invalid Input")

if __name__ == "__main__":
    main()