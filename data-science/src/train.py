# ====================================================
# TRAINING SCRIPT FOR MACHINE FAILURE PREDICTION MODEL
# ====================================================
"""
Trains a Decision Tree Classifier using the training dataset, evaluates performance on the test dataset,
and saves the trained model to the specified output path.
"""

# ================
# IMPORT LIBRARIES
# ================
import argparse
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt

# ======================================
# FUNCTION: PARSE COMMAND-LINE ARGUMENTS
# ======================================
def parse_args():
    '''Parse input arguments for dataset paths, model output, and hyperparameters'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--criterion", type=str, default="gini",
                        help="The function to measure the quality of a split ('gini' or 'entropy')")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="The maximum depth of the tree. If None, then nodes expand until all leaves contain fewer than min_samples_split samples.")

    args = parser.parse_args()
    return args

# ===========================================================
# FUNCTION: TRAIN MODEL, EVALUATE PERFORMANCE, AND SAVE MODEL
# ===========================================================
def main(args):
    '''Read train and test datasets, train model, evaluate model, and save trained model'''

    # ===========================================
    # LOAD TRAIN AND TEST DATASETS FROM CSV FILES
    # ===========================================
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # ===========================================
    # SPLIT THE DATA INTO FEATURES (X) AND TARGET VARIABLE (y)
    # ===========================================
    y_train = train_df["Failure"]  # Target variable
    X_train = train_df.drop(columns=["Failure"])  # Feature variables
    y_test = test_df["Failure"]
    X_test = test_df.drop(columns=["Failure"])

    # ============================================
    # INITIALIZE AND TRAIN THE DECISION TREE MODEL
    # ============================================
    model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # ===================================
    # LOG MODEL HYPERPARAMETERS TO MLFLOW
    # ===================================
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("max_depth", args.max_depth)

    # =============================
    # MAKE PREDICTIONS ON TEST DATA
    # =============================
    yhat_test = model.predict(X_test)

    # ===========================================
    # COMPUTE EVALUATION METRICS AND LOG RESULTS
    # ===========================================
    accuracy = accuracy_score(y_test, yhat_test)  # Compute accuracy score
    print(f"Accuracy of Decision Tree classifier on test set: {accuracy:.2f}")  # Print accuracy

    mlflow.log_metric("Accuracy", float(accuracy))  # Log accuracy to MLflow

    # =================================
    # SAVE TRAINED MODEL TO OUTPUT PATH
    # =================================
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

# =================================================================
# SCRIPT ENTRY POINT: PARSE ARGUMENTS, TRAIN MODEL, AND LOG RESULTS
# =================================================================
if __name__ == "__main__":
    
    mlflow.start_run()

    # ==== PARSE COMMAND-LINE ARGUMENTS ====
    args = parse_args()

    # ==== PRINT ARGUMENT VALUES FOR VERIFICATION ====
    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Criterion: {args.criterion}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    # ==== EXECUTE MAIN TRAINING FUNCTION ====
    main(args)

    mlflow.end_run()
