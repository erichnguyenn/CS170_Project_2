import random

RANDOM_SEED = 42  

#Stub function
def rand_eval(feature_set):
    return random.uniform(0, 100)
    
def forward_selection(total_features):

    curr_features = [] # Start with no features
    
    init_accuracy = rand_eval(curr_features) # Initial state
    
    # Track best found so far
    best_ovr_accuracy = init_accuracy
    best_ovr_features = list(curr_features)
    
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {init_accuracy:.1f}%")
    print("Beginning search.\n")
    
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_add_this_level = -1
        
        # Iterate through all features not in set
        for feature in range(1, total_features + 1):
            if feature not in curr_features:
                temp_features = curr_features + [feature]
                accuracy = rand_eval(temp_features)
                
                print(f"\tUsing feature(s) {set(temp_features)} accuracy is {accuracy:.1f}%")
                
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = feature
        
        # Keep adding even if accuracy decreases (Search all subsets)
        if feature_to_add_this_level != -1:
            curr_features.append(feature_to_add_this_level)
            print(f"\nFeature set {set(curr_features)} was best, accuracy is {best_accuracy_this_level:.1f}%")

            if best_accuracy_this_level > best_ovr_accuracy:
                best_ovr_accuracy = best_accuracy_this_level
                best_ovr_features = list(curr_features)
            else:
                print("(Warning, Accuracy has decreased!)")
                
    print(f"\nFinished search!! The best feature subset is {set(best_ovr_features)}, which has an accuracy of {best_ovr_accuracy:.1f}%")

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
