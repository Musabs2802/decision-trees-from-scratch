def calculate_accuracy(y_predicted, y_actual):
    return sum(y_predicted == y_actual) / len(y_actual)