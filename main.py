import numpy
import pandas
import matplotlib.pyplot as plt1

# Hyperparameters and other controls

load_from_file = False
train_network = True
learning_rate = 0.005
epochs = 300
normalize = True
shuffle = False


def sigmoid(x):
    y = numpy.float128(x)
    return 1 / (1 + numpy.exp(-y))


def mean_squared_error(pred, actual):
    return (actual - pred) ** 2


def train(x_data_frame: pandas.DataFrame, y_series: pandas.Series, w1: numpy.ndarray, w2: numpy.ndarray, alpha=0.01, epochs=10):

    acc = []
    losss = []
    for epoch in range(epochs):
        loss_array = []
        right = []
        true_values = []
        for i in range(len(x_data_frame)):
            x = x_data_frame.values[i:i + 1]

            # Forward Pass

            # hidden
            # (25,) = (8,) ⋅ (8, 25)
            # NOTE: empty strings in the notation mean that dimension is not used.
            # (25,) means a 1 dimensional array vs (25,1) which is a 2-dimensional array with one column
            z1 = x.dot(w1)  # edge going in to the hidden layer

            # (25,)
            a1 = sigmoid(z1)  # edge going out from the hidden layer

            # Output layer
            # (1,) = (25,) ⋅ (25, 1)
            z2 = a1.dot(w2)  # edge going in to the output layer
            out = sigmoid(z2)  # edge going out from the output layer

            true_value = y_series[i + 1]
            pred = out[0][0]
            loss = mean_squared_error(pred, true_value)
            loss_array.append(loss)

            right.append(round(pred) == true_value)
            true_values.append(true_value)


            # Backpropagation

            # Calculate derivative of loss with respect to output
            d_loss_out = 2 * (pred - true_value)

            # Calculate derivative of sigmoid (output layer)
            d_out_z2 = out * (1 - out) # = sigmoid(z2) * (1 - sigmoid(z2))

            # Gradient for w2
            d1 = d_loss_out * d_out_z2
            w2_adj = a1.T.dot(d1)

            # Backpropagate to hidden layer
            d_a1_z1 = a1 * (1 - a1) # = sigmoid(z1) * (1 - sigmoid(z1))
            d1_w2 = d1.dot(w2.T)
            d_loss_z1 = d1_w2 * d_a1_z1

            # Gradient for w1
            w1_adj = x.T.dot(d_loss_z1)

            # Updating parameters
            w1 = w1 - (alpha * w1_adj)
            w2 = w2 - (alpha * w2_adj)

        # evaluate

        right = []
        loss_validation = []
        for i in range(len(x_validate)):
            x = x_validate.values[i:i + 1]
            y = y_validate[y_validate.index[i]]

            prediction_with_confidence = feed_forward(x, w1, w2)
            prediction = round(prediction_with_confidence)
            right.append(prediction == y)
            validation_loss = mean_squared_error(prediction_with_confidence, y)
            loss_validation.append(validation_loss)

        mean_loss = sum(loss_validation) / len(x_validate)
        accuracy = sum(right) / len(right)
        print("epochs:", epoch + 1, "======== acc:", accuracy, "---- right:", sum(right), "/",len(right))

        acc.append(accuracy)
        losss.append(mean_loss)

    return (acc, losss, w1, w2)


def feed_forward(x: pandas.Series, w1: numpy.ndarray, w2: numpy.ndarray):
    """
    Returns a fractional prediction between 0 and 1.
    The closer to 0 or 1 the more confident the prediction
    """

    # hidden
    # (25,) = (8,) ⋅ (8, 25)
    # NOTE: empty strings in the notation mean that dimension is not used.
    # (25,) means a 1 dimensional array vs (25,1) which is a 2-dimensional array with one column
    z1 = x.dot(w1)  # edge going in to the hidden layer

    # (25,)
    a1 = sigmoid(z1)  # edge going out from the hidden layer

    # Output layer
    # (1,) = (25,) ⋅ (25, 1)
    z2 = a1.dot(w2)  # edge going in to the output layer
    out = sigmoid(z2)  # edge going out from the output layer
    return out[0][0]


# data preparation

def prepare_data():
    # thanks to https://www.kaggle.com/code/mkulio/titanic-data-preparation for preparation code

    path_train = "input/titanic/train.csv"
    path_test = "input/titanic/test.csv"

    train_data = convert_to_data_frames(path_train)
    test_data = convert_to_data_frames(path_test)

    all_data = pandas.concat([train_data, test_data])

    all_data = make_ticket_column_numeric(all_data)

    # to simplify things for this project:

    # drop the Name column
    all_data.drop("Name", axis=1, inplace=True)

    # convert unknown ages to -1
    all_data['Age'] = all_data['Age'].fillna(-1)

    # add the missing Fare value
    all_data.loc[1044, 'Fare'] = 8

    # add the missing Embarked values
    all_data.loc[62, 'Embarked'] = 'C'
    all_data.loc[830, 'Embarked'] = 'C'

    pandas.set_option('display.max_columns', None)

    # remove the Cabin column
    all_data.drop("Cabin", axis=1, inplace=True)

    all_data = convert_categories_to_numbers(all_data, "Sex")
    all_data = convert_categories_to_numbers(all_data, "Embarked")

    # all_data.to_csv("all_data.csv", sep='\t')

    return all_data.iloc[:891, :], all_data.iloc[891:, :]


def convert_to_data_frames(path: str):
    data = pandas.read_csv(path)
    data.set_index('PassengerId', inplace=True)
    return data


def make_ticket_column_numeric(all_data: pandas.DataFrame):
    # make the ticket numbers numeric
    ticket = all_data['Ticket'].str.split()
    ticket_numbers = ticket.str.get(-1)

    # create a new column called Ticket Number that contains the numeric ticket numbers
    all_data['Ticket Number'] = ticket_numbers

    # manually create ticket numbers for those that are missing
    all_data.loc[180, 'Ticket Number'] = 0
    all_data.loc[272, 'Ticket Number'] = 1
    all_data.loc[303, 'Ticket Number'] = 2
    all_data.loc[598, 'Ticket Number'] = 3

    # convert column from string to number
    try:
        all_data['Ticket Number'] = pandas.to_numeric(all_data['Ticket Number'])
    except ValueError as e:
        print(e)

    all_data.drop("Ticket", axis=1, inplace=True)

    return all_data


def convert_categories_to_numbers(data: pandas.DataFrame, column_name):
    # convert a category column to numbers
    # thanks to https://stackoverflow.com/a/61761109
    data[[column_name]] = data[[column_name]].apply(lambda col: pandas.Categorical(col).codes)
    return data



if __name__ == '__main__':

    x_train, x_test = prepare_data()

    if shuffle:
        x_train = x_train.sample(frac = 1)

    y_train = x_train.pop("Survived")
    x_test = x_test.drop(columns=["Survived"])

    if normalize:
        df_z_scaled = x_train

        # apply normalization techniques
        # thanks https://www.geeksforgeeks.org/data-normalization-with-pandas/
        for column in df_z_scaled.columns:
            df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()
        x_train = df_z_scaled

    number_of_hidden_neurons = 25
    number_of_features = x_train.shape[1]
    training_examples = x_train.shape[0]

    numpy.random.seed(0)

    if load_from_file:
        weights_between_input_and_hidden = numpy.loadtxt('w1.txt', dtype=int)
        weights_between_hidden_and_output = numpy.reshape((numpy.loadtxt('w2.txt', dtype=int)), (-1, 1))
    else:
        weights_between_input_and_hidden = numpy.random.uniform(-1, 1, (number_of_features, number_of_hidden_neurons))
        weights_between_hidden_and_output = numpy.random.uniform(-1, 1, (number_of_hidden_neurons, 1))

    x_training = x_train.iloc[:800, :]
    x_validate = x_train.iloc[800:, :]

    y_training = y_train.iloc[:800, ]
    y_validate = y_train.iloc[800:, ]

    if train_network:

        # train

        acc, losss, weights_between_input_and_hidden, weights_between_hidden_and_output = train(x_training, y_training, weights_between_input_and_hidden, weights_between_hidden_and_output,
                                   learning_rate, epochs)

        numpy.savetxt('w1.txt', weights_between_input_and_hidden, fmt='%d')
        numpy.savetxt('w2.txt', weights_between_hidden_and_output, fmt='%d')
        # b = numpy.loadtxt('test1.txt', dtype=int)

        # plotting accuracy
        plt1.figure()
        plt1.plot(acc)
        plt1.ylabel('Accuracy')
        plt1.xlabel("Epochs:")
        plt1.savefig("accuracy.png")

        # plotting Loss
        plt1.figure()
        plt1.plot(losss)
        plt1.ylabel('Loss')
        plt1.xlabel("Epochs:")
        plt1.savefig("loss.png")

    else:
        y_preds = []
        print("PassengerId,Survived")
        for i in range(len(x_test)):
            x = x_test.values[i:i + 1]
            y_pred = feed_forward(x, weights_between_input_and_hidden, weights_between_hidden_and_output)
            y_pred_rounded = round(y_pred)
            print(f"{x_test.index[i]},{y_pred_rounded}")
            y_preds.append(y_pred_rounded)
        print(f"{sum(y_preds) / len(y_preds)}")