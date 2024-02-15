# main.py
# reads in data, splits into train, validation, and test data
# builds 3 decision tree models
# determines best model using cross validation and tests it on test data

import pandas as pd
from datasplit import split_data, split_4_folds, calculate_accuracy
import decisiontree

main_df = pd.read_csv("car.data", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
print(main_df.head())

train_data, test_data = split_data(main_df, 0.6)

train_data_1, val_data_1, train_data_2, val_data_2, train_data_3, val_data_3, train_data_4, val_data_4 = split_4_folds(train_data)

constant = 1
average_train_samples = (train_data_1.shape[0] + train_data_2.shape[0] + train_data_3.shape[0] + train_data_4.shape[0]) / 4.0

print("Training Trees with gain threshold of 0.1")

tree_1 = decisiontree.DecisionTree(0.1)
tree_1.train(train_data_1)
val_error_1 = 1 - calculate_accuracy(tree_1.test(val_data_1))
print("Tree 1: " + str(tree_1.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_1.test(train_data_1))))
print("Validation Error: " + str(val_error_1))

tree_2 = decisiontree.DecisionTree(0.1)
tree_2.train(train_data_2)
val_error_2 = 1 - calculate_accuracy(tree_2.test(val_data_2))
print("Tree 2: " + str(tree_2.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_2.test(train_data_2))))
print("Validation Error: " + str(val_error_2))

tree_3 = decisiontree.DecisionTree(0.1)
tree_3.train(train_data_3)
val_error_3 = 1 - calculate_accuracy(tree_3.test(val_data_3))
print("Tree 3: " + str(tree_3.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_3.test(train_data_3))))
print("Validation Error: " + str(val_error_3))

tree_4 = decisiontree.DecisionTree(0.1)
tree_4.train(train_data_4)
val_error_4 = 1 - calculate_accuracy(tree_4.test(val_data_4))
print("Tree 4: " + str(tree_4.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_4.test(train_data_4))))
print("Validation Error: " + str(val_error_4))

average_leaf_nodes = (tree_1.leaf_node_count + tree_2.leaf_node_count + tree_3.leaf_node_count + tree_4.leaf_node_count) / 4.0
average_val_error = (val_error_1 + val_error_2 + val_error_3 + val_error_4) / 4.0

generalization_error_1 = average_val_error + constant * (average_leaf_nodes / average_train_samples)
print("\nGeneralization Error of Decision Tree with 0.1 gain threshold: " + str(generalization_error_1))

print("\n\nTraining Trees with gain threshold of 0.05")
train_data_1, val_data_1, train_data_2, val_data_2, train_data_3, val_data_3, train_data_4, val_data_4 = split_4_folds(train_data)

tree_1 = decisiontree.DecisionTree(0.05)
tree_1.train(train_data_1)
val_error_1 = 1 - calculate_accuracy(tree_1.test(val_data_1))
print("Tree 1: " + str(tree_1.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_1.test(train_data_1))))
print("Validation Error: " + str(val_error_1))

tree_2 = decisiontree.DecisionTree(0.05)
tree_2.train(train_data_2)
val_error_2 = 1 - calculate_accuracy(tree_2.test(val_data_2))
print("Tree 2: " + str(tree_2.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_2.test(train_data_2))))
print("Validation Error: " + str(val_error_2))

tree_3 = decisiontree.DecisionTree(0.05)
tree_3.train(train_data_3)
val_error_3 = 1 - calculate_accuracy(tree_3.test(val_data_3))
print("Tree 3: " + str(tree_3.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_3.test(train_data_3))))
print("Validation Error: " + str(val_error_3))

tree_4 = decisiontree.DecisionTree(0.05)
tree_4.train(train_data_4)
val_error_4 = 1 - calculate_accuracy(tree_4.test(val_data_4))
print("Tree 4: " + str(tree_4.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_4.test(train_data_4))))
print("Validation Error: " + str(val_error_4))

average_leaf_nodes = (tree_1.leaf_node_count + tree_2.leaf_node_count + tree_3.leaf_node_count + tree_4.leaf_node_count) / 4.0
average_val_error = (val_error_1 + val_error_2 + val_error_3 + val_error_4) / 4.0

generalization_error_2 = average_val_error + constant * (average_leaf_nodes / average_train_samples)
print("\nGeneralization Error of Decision Tree with 0.05 gain threshold: " + str(generalization_error_2))

print("\n\nTraining Trees with 0.15 gain threshold")
train_data_1, val_data_1, train_data_2, val_data_2, train_data_3, val_data_3, train_data_4, val_data_4 = split_4_folds(train_data)

tree_1 = decisiontree.DecisionTree(0.15)
tree_1.train(train_data_1)
val_error_1 = 1 - calculate_accuracy(tree_1.test(val_data_1))
print("Tree 1: " + str(tree_1.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_1.test(train_data_1))))
print("Validation Error: " + str(val_error_1))

tree_2 = decisiontree.DecisionTree(0.15)
tree_2.train(train_data_2)
val_error_2 = 1 - calculate_accuracy(tree_2.test(val_data_2))
print("Tree 2: " + str(tree_2.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_2.test(train_data_2))))
print("Validation Error: " + str(val_error_2))

tree_3 = decisiontree.DecisionTree(0.15)
tree_3.train(train_data_3)
val_error_3 = 1 - calculate_accuracy(tree_3.test(val_data_3))
print("Tree 3: " + str(tree_3.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_3.test(train_data_3))))
print("Validation Error: " + str(val_error_3))

tree_4 = decisiontree.DecisionTree(0.15)
tree_4.train(train_data_4)
val_error_4 = 1 - calculate_accuracy(tree_4.test(val_data_4))
print("Tree 4: " + str(tree_4.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(tree_4.test(train_data_4))))
print("Validation Error: " + str(val_error_4))

average_leaf_nodes = (tree_1.leaf_node_count + tree_2.leaf_node_count + tree_3.leaf_node_count + tree_4.leaf_node_count) / 4.0
average_val_error = (val_error_1 + val_error_2 + val_error_3 + val_error_4) / 4.0

generalization_error_3 = average_val_error + constant * (average_leaf_nodes / average_train_samples)
print("\nGeneralization Error of Decision Tree with 0.15 gain threshold: " + str(generalization_error_2))

if generalization_error_1 < generalization_error_2 and generalization_error_1 < generalization_error_3:
    print("\nGeneralization Error Lowest for Decision Tree with 0.1 gain threshold.")
    final_tree = decisiontree.DecisionTree(0.1)
elif generalization_error_2 < generalization_error_1 and generalization_error_2 < generalization_error_3:
    print("\nGeneralization Error Lowest for Decision Tree with 0.05 gain threshold.")
    final_tree = decisiontree.DecisionTree(0.05)
else:
    print("\nGeneralization Error Lowest for Decision Tree with 0.15 gain threshold.")
    final_tree = decisiontree.DecisionTree(0.15)

print("Training Decision Tree with least generalization error on full training data set")

final_tree.train(train_data)
test_error = 1 - calculate_accuracy(final_tree.test(test_data))
print("Final Tree: " + str(final_tree.node_count) + " nodes")
print("Training Error: " + str(1.0 - calculate_accuracy(final_tree.test(train_data))))
print("Test Error: " + str(val_error_1))

print("\nDone.")
