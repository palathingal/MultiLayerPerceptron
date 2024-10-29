from mlp import Mlp
import random
import pandas as pd
import numpy as np
import csv

def ocr(training_population=5000, testing_population=1000):
    print("Loading data...")
    train = pd.read_csv("mnist_train.csv")
    train = process_df(train)
    test_set = pd.read_csv("mnist_test.csv")
    test_set = process_df(test_set)
    print("Loaded {} rows for training.".format(train.shape[0]))
    print("Loaded {} rows for testing.".format(test_set.shape[0]))
    nn = Mlp(init_nodes=784, learning_rate=.05)
    nn.add_layer(300)
    nn.add_layer(150, function="relu")
    nn.add_layer(10)

    print("Training the network with {} samples...".format(training_population))
    for i in range(training_population):
        data = train.sample(n=1)
        label = data["label"].tolist()[0]
        inputs = list(data.iloc[0, 1:])
        outputs = [0] * 10
        outputs[label] = 1
        nn.train(inputs, outputs)

    print("Trained successfully.")
    # nn.save("ocr.mlp")

    weights_biases = nn.get_weights_and_biases()


    print("Testing with {} samples...".format(testing_population))
    c_m = np.zeros(shape=(10, 10))
    for i in range(testing_population):
        data = test_set.sample(n=1)
        inputs = list(data.iloc[0, 1:])
        label = data["label"].tolist()[0]
        out_class, out_prob = nn.predict(inputs)
        c_m[label][out_class] += 1

    print("Results:")

    correct_guesses = np.sum(np.diagonal(c_m))
    total_guesses = c_m.sum()
    accuracy = correct_guesses / total_guesses

    recall = 0
    precision = 0
    c_m_t = c_m.T

    for i in range(10):
        correct_guesses = c_m[i][i]
        total_row = np.sum(c_m[i])
        total_col = np.sum(c_m_t[i])
        recall += (correct_guesses / total_row) if total_row > 0 else 0
        precision += (correct_guesses / total_col) if total_col > 0 else 0

    recall = recall / 10
    precision = precision / 10

    print("\tRecall: {0:.2f}\n\tPrecision: {0:.2f}\n\tAccuracy: {0:.2f}".format(recall, precision, accuracy))

    # Return weights_biases from the ocr function
    return weights_biases


def filter_pixel(x):
    return x / 255

def float_to_hex_8bit(f):
    """
    Converts a floating-point number to an 8-bit unsigned integer and then to hexadecimal format.
    Scales the float to the range [0, 255].
    """
    # Scale the float to the range [0, 255]
    int_value = np.clip(int(np.round(f * 255)), 0, 255)  # Scale to 0 to 255
    # Convert to hexadecimal, ensure it's formatted to 2 characters
    hex_value = format(int_value, '02x')  # Convert to 2-digit hex
    return hex_value


def process_df(df):
    labels = df["label"]
    df = df.drop(["label"], axis=1)
    df = df.apply(np.vectorize(filter_pixel))
    df = pd.concat([labels, df], axis=1)
    return df

def save_weights_and_biases(weights_biases, filename="weights_biases.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Layer", "Weights", "Biases"])
        for i, (weights, biases) in enumerate(zip(weights_biases['weights'], weights_biases['biases'])):
            writer.writerow([f"Layer {i+1}", weights.tolist(), biases.tolist()])

def save_weights_and_biases_to_hex_8bit(weights_biases, filename="weights_biases_8bit.txt"):
    with open(filename, mode='w') as file:
        for i, (weights, biases) in enumerate(zip(weights_biases['weights'], weights_biases['biases'])):
            file.write(f"# Weights of Layer {i+1}\n")
            for weight_matrix in weights:
                for weight in weight_matrix:
                    file.write(float_to_hex_8bit(weight) + '\n')

            file.write(f"# Biases of Layer {i+1}\n")
            for bias in biases:
                for b in bias:
                    file.write(float_to_hex_8bit(b) + '\n')


weights_biases = ocr(training_population=60000, testing_population=10000)

save_weights_and_biases_to_hex_8bit(weights_biases)
