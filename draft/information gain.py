import math
from collections import Counter


def entropy(class_counts):
    """计算熵"""
    total = sum(class_counts.values())
    return -sum((count / total) * math.log2(count / total) for count in class_counts.values() if count > 0)


def information_gain(data, attribute, target_attribute):
    """计算指定属性的信息增益"""
    target_counts = Counter(record[target_attribute] for record in data)
    total_entropy = entropy(target_counts)

    attribute_values = set(record[attribute] for record in data)
    weighted_entropy = 0.0
    total_size = len(data)

    for value in attribute_values:
        subset = [record for record in data if record[attribute] == value]
        subset_counts = Counter(record[target_attribute] for record in subset)
        subset_entropy = entropy(subset_counts)
        weighted_entropy += (len(subset) / total_size) * subset_entropy

    return total_entropy - weighted_entropy


def choose_best_attribute(data, attributes, target):
    """选择信息增益最大的属性"""
    gains = {attr: information_gain(data, attr, target) for attr in attributes}
    return max(gains, key=gains.get)


def build_tree(data, attributes, target):
    """递归构建决策树"""
    target_values = [record[target] for record in data]
    if len(set(target_values)) == 1:
        return target_values[0]  # 纯类别，不需要继续分裂
    if not attributes:
        return Counter(target_values).most_common(1)[0][0]  # 返回最多的类别

    best_attr = choose_best_attribute(data, attributes, target)
    tree = {best_attr: {}}

    for value in set(record[best_attr] for record in data):
        subset = [record for record in data if record[best_attr] == value]
        new_attributes = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = build_tree(subset, new_attributes, target)

    return tree


def print_tree(tree, indent=""):
    """打印决策树"""
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f"{indent}{key}")
            for subkey, subtree in value.items():
                print(f"{indent}  ├── {subkey}:")
                print_tree(subtree, indent + "  │   ")
    else:
        print(f"{indent}  └── [Class: {tree}]")


# 新数据集
weather_dataset = [
    {'Weather': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play?': 'No'},
    {'Weather': 'Cloudy', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'Play?': 'Yes'},
    {'Weather': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play?': 'Yes'},
    {'Weather': 'Cloudy', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play?': 'Yes'},
    {'Weather': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play?': 'No'},
    {'Weather': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'Play?': 'No'},
    {'Weather': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'Play?': 'Yes'},
    {'Weather': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'Play?': 'No'},
    {'Weather': 'Cloudy', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Wind': 'Weak', 'Play?': 'Yes'},
    {'Weather': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Strong', 'Play?': 'No'},
]

# 构建决策树
attributes = ['Weather', 'Temperature', 'Humidity', 'Wind']
target = 'Play?'
decision_tree = build_tree(weather_dataset, attributes, target)

# 打印决策树
print("决策树结构:")
print_tree(decision_tree)
