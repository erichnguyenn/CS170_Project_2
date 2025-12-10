import random
import math
import time

RANDOM_SEED = 42  

def rand_eval(feature_set):
    return random.uniform(0, 100)
    
def forward_selection(X, y):
    total_features = len(X[0])
    current_features = [] #start with empty set

    # Evaluate accuracy with no features
    initial_accuracy = evaluate_subset(X, y, current_features)

    # Keep track of best overall
    best_overall_accuracy = initial_accuracy
    best_overall_features = list(current_features)

    print(f"Running nearest neighbor with no features (default rate), "
          f"using \"leaving-one-out\" evaluation, I get an accuracy of {initial_accuracy * 100:.1f}%")
    print("Beginning search.\n")

    # Iterate to add features
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_add_this_level = -1

        # Try adding each feature not already in the set
        for feature in range(1, total_features + 1):
            if feature not in current_features:
                temp_features = current_features + [feature]
                accuracy = evaluate_subset(X, y, temp_features)

                print(f"\tUsing feature(s) {set(temp_features)} accuracy is {accuracy * 100:.1f}%")

                # Check if best so far this level
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = feature

        # Add the best feature found this level
        if feature_to_add_this_level != -1:
            current_features.append(feature_to_add_this_level)
            print(f"\nFeature set {set(current_features)} was best, "
                  f"accuracy is {best_accuracy_this_level * 100:.1f}%")

            # Update overall best if improved
            if best_accuracy_this_level > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_features = list(current_features)
            else:
                print("(Warning, Accuracy has decreased!)")

    print(f"\nFinished search!! The best feature subset is {set(best_overall_features)}, "
          f"which has an accuracy of {best_overall_accuracy * 100:.1f}%")

def backward_elimination(X, y):
    total_features = len(X[0])
    current_features = list(range(1, total_features + 1))

    # Evaluate accuracy with all features
    initial_accuracy = evaluate_subset(X, y, current_features)

    # Keep track of best overall
    best_overall_accuracy = initial_accuracy
    best_overall_features = list(current_features)

    print(f"Running nearest neighbor with all features, "
          f"using \"leaving-one-out\" evaluation, I get an accuracy of {initial_accuracy * 100:.1f}%")
    print("Beginning search.\n")

    # Iterate to remove features
    for i in range(total_features):
        best_accuracy_this_level = -1
        feature_to_remove_this_level = -1

        for feature in current_features:
            temp_features = [f for f in current_features if f != feature]
            accuracy = evaluate_subset(X, y, temp_features)

            print(f"\tUsing feature(s) {set(temp_features)} accuracy is {accuracy * 100:.1f}%")

            # Check if best so far this level
            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                feature_to_remove_this_level = feature

        # Remove the best feature found this level
        if feature_to_remove_this_level != -1:
            current_features.remove(feature_to_remove_this_level)
            print(f"\nFeature set {set(current_features)} was best, "
                  f"accuracy is {best_accuracy_this_level * 100:.1f}%")

            # Update overall best if improved
            if best_accuracy_this_level >= best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_level
                best_overall_features = list(current_features)
            else:
                print("(Warning, Accuracy has decreased!)")

    print(f"\nFinished search!! The best feature subset is {set(best_overall_features)}, "
          f"which has an accuracy of {best_overall_accuracy * 100:.1f}%")

# ---------------- Part II: NN classifier + leave-one-out validator ----------------

def load_dataset(path):
    X = []
    y = []

    # Read data
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

        for i in range(len(self.train_X)):
            train_instance = self.train_X[i]
            dist_sq = 0.0
            for j in range(len(train_instance)):
                diff = train_instance[j] - x[j]
                dist_sq += diff * diff

            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_label = self.train_y[i]

        return best_label
    
def default_rate(y):
    """Accuracy using the majority class only."""
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / len(y)

    
def leave_one_out_accuracy(X, y, feature_subset, verbose=True, print_summary=True):

    if not feature_subset:
        # For Part III, allow empty subset using default rate
        acc = default_rate(y)
        if print_summary:
            print(f"Accuracy: {acc * 100:.1f}%")
        return acc

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
            row = [X[k][j] for j in feat_idx] # select chosen features
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

    if print_summary:
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

    # Convert input to list of integers
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

def evaluate_subset(X, y, subset):

    if not subset:
        return default_rate(y)
    return leave_one_out_accuracy(X, y, subset, verbose=False, print_summary=False)


def main():

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("Welcome to Eric, Rafat, and Lynvy's Feature Selection Algorithm.")

    print("\nType the number of the algorithm you want to run.\n")
    print("1) Forward Selection ")
    print("2) Backward Elimination ")
    print("3) Nearest Neighbor Validator ")

    choice = input("\nEnter choice [1-3]: ").strip()

    if choice in ("1", "2"):
        dataset_path = input("\nType in the name of the file to test: ").strip()

        try:
            X, y = load_dataset(dataset_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        print(f"\nThis dataset has {len(X[0])} features (not including the class attribute), "
              f"with {len(X)} instances.")
        print("Please wait while I normalize the data… Done!")

        if choice == "1":
            forward_selection(X, y)
        else:
            backward_elimination(X, y)

    elif choice == "3":
        run_part2()
    else:
        print("Invalid Input")
if __name__ == "__main__":
    main()
    

# Lynvy Chang-lchan171-Session 1, Eric Nguyen-enguy197-Session 1, Rafat Alam-ralam016-Session 1
#DatasetID: 2
#Small Dataset Results:
#- Forward: Feature Subset: {3, 5}, Acc: 92%
#- Backward: Feature Subset: {2, 4, 5, 7, 10}, Acc: 82%
#Large Dataset Results:
#- Forward: Feature Subset: {1, 27}, Acc:95.5%
#- Backward: Feature Subset: {27}, Acc: 84.7%
#titanic Dataset Results:
#- Forward: Feature Subset: {2}, Acc: 78.0%
#- Backward: Feature Subset: {2}, Acc: 78.0%