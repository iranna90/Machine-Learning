from neural_net import mnist_wrapper, Network
from matplotlib import pyplot as plt, cm


def print_details(data):
    input_data, output = data
    print_label(output)
    plot_image(input_data)


def plot(pattern_before, pattern_after, number_of_patterns):
    f, axarr = plt.subplots(4, 2)
    axarr[0, 0].imshow(pattern_before[0][0].reshape(28, 28))
    axarr[0, 1].imshow(pattern_after[0][0].reshape(28, 28))
    axarr[1, 0].imshow(pattern_before[0][29].reshape(28, 28))
    axarr[1, 1].imshow(pattern_after[0][29].reshape(28, 28))
    axarr[2, 0].imshow(pattern_before[1][0].reshape(5, 6))
    axarr[2, 1].imshow(pattern_after[1][0].reshape(5, 6))
    axarr[3, 0].imshow(pattern_before[1][9].reshape(5, 6))
    axarr[3, 1].imshow(pattern_after[1][9].reshape(5, 6))
    plt.show()


def method_name(number_of_patterns, pattern_after, pattern_before):
    f, axarr = plt.subplots(number_of_patterns, 2)
    row = 0
    for before, after in zip(pattern_before[0], pattern_after[0]):
        axarr[row, 0].imshow(before.reshape(28, 28))
        axarr[row, 1].imshow(after.reshape(28, 28))
        row += 1
    for before, after in zip(pattern_before[1], pattern_after[1]):
        axarr[row, 0].imshow(before.reshape(5, 6))
        axarr[row, 1].imshow(after.reshape(5, 6))
        row += 1

    plt.show()


def plot_image(input_data):
    result = input_data.reshape((28, 28))
    plt.imshow(result, cmap=cm.gray)
    plt.show()


def print_label(output):
    for index, value in enumerate(output):
        if value == 1:
            print "label is :{}".format(index)
            break


training_data, validation_data, test_data = mnist_wrapper.load_data_wrapper()
neurons_per_layer = [784, 30, 10]
print "started training"
net = Network(neurons_per_layer)
# before patterns
weight_patterns_before = []
for w_layer in net.weights:
    weight_patterns_before.append(w_layer.copy())

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
print "training completed"
# before patterns
weight_patterns_after = []
for w_layer in net.weights:
    weight_patterns_after.append(w_layer.copy())

plot(weight_patterns_before, weight_patterns_after, 30 + 10)
