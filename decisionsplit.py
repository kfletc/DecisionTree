# File: decisionsplit.py
# has methods necessary when given input data for determining the best split of that data

import math

# calculates entropy from given data
def entropy(data):
    if data.shape[0] == 0:
        return 0
    classes = data['class'].drop_duplicates()
    summation = 0.0
    total = data.shape[0]
    for c in classes:
        data_class = data[data['class'] == c]
        class_num = data_class.shape[0]
        p_class = float(class_num) / float(total)
        expression = p_class * math.log(p_class, 2)
        summation = summation + expression
    return -1*summation

# calculates gain of a given split applied to given data
def gain(rule_attribute, rule, data):
    initial_entropy = entropy(data)
    total = data.shape[0]
    rule_true = data[data[rule_attribute] == rule]
    rule_false = data[data[rule_attribute] != rule]
    true_count = rule_true.shape[0]
    false_count = rule_false.shape[0]
    true_weight = float(true_count) / float(total)
    false_weight = float(false_count) / float(total)
    true_entropy = entropy(rule_true)
    false_entropy = entropy(rule_false)
    weighted_entropy = true_weight*true_entropy + false_weight*false_entropy
    return initial_entropy - weighted_entropy

# returns the split that yields the highest gain when given a set of possible rules and data
def best_gain(possible_rules, data):
    columns = data.columns
    max_gain = 0.0
    best_attribute = None
    best_rule = None
    for col in columns:
        if col != 'class' and possible_rules[col] is not None:
            for rule in possible_rules[col]:
                current_gain = gain(col, rule, data)
                # print("attribute: " + col + " rule: " + rule + " gain: " + str(current_gain))
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_attribute = col
                    best_rule = rule
    return best_attribute, best_rule, max_gain

# looks at given data to determine all possible single attribute splits
def get_all_splits(data):
    columns = data.columns
    possible_rules = {}
    for col in columns:
        if col != 'class':
            values = data[col].drop_duplicates()
            if values.shape[0] <= 1:
                possible_rules[col] = None
            else:
                possible_rules[col] = values.tolist()
    return possible_rules

# gets the best single attribute split according to maximizing gain from the given data
def get_best_split(data):
    possible_rules = get_all_splits(data)
    best_attribute, best_rule, rule_gain = best_gain(possible_rules, data)
    return best_attribute, best_rule, rule_gain
