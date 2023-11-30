# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

import numpy
import pandas

def prepare_data_at(path: str):

    # thanks to https://www.kaggle.com/code/mkulio/titanic-data-preparation for preparation code

    data = pandas.read_csv(path)
    data.head(3)

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


def prepare_name(all_data: pandas.DataFrame):
    # split name out from the Name column
    # results in ["family name", "title & first"]
    family = all_data["Name"].str.split(",")

    # add the Family Name to the data
    all_data["Family Name"] = family.str.get(0)

    print(all_data.head(3))

    # split title out
    title = family.str.get(1).str.split(".").str.get(0)
    print(title)

def convert_categories_to_numbers(data: pandas.DataFrame, column_name):

    # convert a category column to numbers
    # thanks to https://stackoverflow.com/a/61761109
    data[[column_name]] = data[[column_name]].apply(lambda col: pandas.Categorical(col).codes)
    return data

if __name__ == '__main__':

    path_train = "input/titanic/train.csv"
    path_test = "input/titanic/test.csv"

    train_data = prepare_data_at(path_train)
    test_data = prepare_data_at(path_test)

    all_data = pandas.concat([train_data, test_data])
    # print(train_data.shape, test_data.shape)
    # print(all_data.shape)
    #
    # print(all_data.tail(3))
    #
    # all_data.isnull().sum()
    #
    #
    # age = pandas.cut(train_data['Age'], [0, 2, 5, 10, 12, 18, 80])
    # train_data.pivot_table('Survived', [age])

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

    print(all_data.head(3))
    print(all_data.isnull().sum())

    all_data.to_csv("all_data.csv", sep='\t')
