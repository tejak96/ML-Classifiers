# coding=utf-8
import csv
import os
script_dir = os.path.dirname(__file__)
from sklearn.model_selection import train_test_split
full_path = os.path.join(script_dir, '../dataset/data.csv')

def get_data():
        features_train=[]
        features_test=[]
        labels_train=[]
        labels_test=[]
        features=[]
        labels=[]
        with open(full_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                features.append(row[0])
                labels.append(row[1])
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

        return features_train, features_test, labels_train, labels_test