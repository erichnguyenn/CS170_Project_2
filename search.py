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

# ---------------- Part II: NN classifier + leave-one-out validator ----------------

def load_dataset(path):
    X = []
    y = []

    # Read each line and parse label + features
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(float(parts[0]))
            features = [float(v) for v in parts[1:]]
            y.append(label)
            X.append(features)

    if not X:
        raise ValueError("Dataset appears to be empty.")

    n = len(X)
    d = len(X[0])

    # Compute mean and std for each feature (for normalization)
    means = [0.0] * d
    stds = [0.0] * d

    # Means
    for j in range(d):
        col_sum = 0.0
        for i in range(n):
            col_sum += X[i][j]
        means[j] = col_sum / n

    # Standard deviations
    for j in range(d):
        var_sum = 0.0
        for i in range(n):
            diff = X[i][j] - means[j]
            var_sum += diff * diff
        variance = var_sum / n
        stds[j] = math.sqrt(variance)
        if stds[j] == 0.0:
            stds[j] = 1.0  # avoid division by zero for constant features

    # Normalize (z-score)
    X_norm = []
    for i in range(n):
        row = []
        for j in range(d):
            z = (X[i][j] - means[j]) / stds[j]
            row.append(z)
        X_norm.append(row)

    return X_norm, y


class NNClassifier:

    def __init__(self):
        self.train_X = None
        self.train_y = None

    def Train(self, X, y):
        """Store training data."""
        self.train_X = X
        self.train_y = y

    def Test(self, x):
        """
        Predict the class label for a single test instance x.
        Uses Euclidean distance to all training instances.
        """
        if self.train_X is None or self.train_y is None:
            raise ValueError("Classifier has not been trained yet.")

        best_dist_sq = None
        best_label = None

        # Compare x to every training row
        for i in range(len(self.train_X)):
            train_instance = self.train_X[i]
            dist_sq = 0.0
            # Compute squared Euclidean distance
            for j in range(len(train_instance)):
                diff = train_instance[j] - x[j]
                dist_sq += diff * diff

            # Update nearest neighbor if this distance is smaller
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_label = self.train_y[i]

        return best_label
    
def leave_one_out_accuracy(X, y, feature_subset, verbose=True):
    
    if not feature_subset:
        raise ValueError("Feature subset must not be empty.")

    # Convert to 0-based indices
    feat_idx = [f - 1 for f in feature_subset]

    n = len(X)
    correct = 0

    start_time = time.time()

    for i in range(n):
        # Build training set (all but i)
        X_train = []
        y_train = []
        for k in range(n):
            if k == i:
                continue
            row = [X[k][j] for j in feat_idx]  # select chosen features
            X_train.append(row)
            y_train.append(y[k])

        # Test instance (only chosen features)
        x_test = [X[i][j] for j in feat_idx]
        y_true = y[i]

        clf = NNClassifier()
        clf.Train(X_train, y_train)
        y_pred = clf.Test(x_test)

        if y_pred == y_true:
            correct += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} instances...")

    elapsed = time.time() - start_time
    accuracy = correct / n

    print(f"\nLeave-one-out took {elapsed:.2f} seconds "
          f"for {n} instances using features {set(feature_subset)}.")
    print(f"Correctly classified {correct} out of {n} instances.")
    print(f"Accuracy: {accuracy * 100:.1f}%")

    return accuracy

def run_part2():
    print("\n--- Part II: Nearest Neighbor Classifier + Leave-One-Out ---")

    dataset_path = input(
        "\nPlease enter the dataset file name "
        "(e.g., small-test-dataset.txt or large-test-dataset.txt): "
    ).strip()

    try:
        X, y = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    num_instances = len(X)
    num_features = len(X[0])

    print(f"\nLoaded dataset '{dataset_path}'.")
    print(f"Number of instances: {num_instances}")
    print(f"Number of features: {num_features}")

    print("\nEnter the feature subset as space-separated indices.")
    print("  Examples:")
    print("    For the small dataset test: 3 5 7")
    print("    For the large dataset test: 1 15 27")
    raw = input("Feature subset: ").strip()

    try:
        feature_subset = [int(tok) for tok in raw.split()]
        if not feature_subset:
            raise ValueError
    except ValueError:
        print("Invalid feature subset. Please enter integers like: 3 5 7")
        return

    # Check that all features are in range
    invalid = [f for f in feature_subset if f < 1 or f > num_features]
    if invalid:
        print(f"Invalid feature indices (must be between 1 and {num_features}): {invalid}")
        return

    acc = leave_one_out_accuracy(X, y, feature_subset, verbose=True)
    print(f"\nFinal accuracy with features {set(feature_subset)}: {acc * 100:.1f}%\n")


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
    print("1) Forward Selection (Part I)")
    print("2) Backward Elimination (Part I)")
    print("3) Nearest Neighbor Validator (Part II)")

    choice = input("\nEnter choice [1-2]: ").strip()
    if choice == "1":
        forward_selection(total_features)
    elif choice == "2":
        backward_elimination(total_features)
    elif choice == "3":
        run_part2()
    else:
        print("Invalid Input")

if __name__ == "__main__":
    main()
