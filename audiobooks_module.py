# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf


class audiobooks_model:

    # import model and scaler-files
    def __init__(self, model_file, scaler_file):
        with open(scaler_file, 'rb') as scaler:
            self.scaler = pickle.load(scaler)
            self.data = None
        self.model = tf.keras.models.load_model(model_file)

    # function to load and clean new data to get it in the form the model expects
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file)

        # drop first and last column
        df = df.drop(df.columns[[0, -1]], axis=1)

        self.preprocessed_data = df.copy()

        # scale the new data
        self.data = self.scaler.transform(df)

    # function that returns the probability of a customer buying again
    def predicted_probability(self):
        if self.data is not None:
            pred = self.model.predict(self.data)[:, 1]
            return pred

    # function that returns the predicted class 0=wont buy, 1=will buy
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = []
            for i in self.model.predict(self.data):
                if i[0] > 0.5:
                    pred_outputs.append(0)
                else:
                    pred_outputs.append(1)
            return pred_outputs

    # function that adds the functionality of the two functions above but also adds them as
    # new columns to the dataframe before returning it
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.model.predict(self.data)[:, 1]
            pred_outputs = []
            for i in self.model.predict(self.data):
                if i[0] > 0.5:
                    pred_outputs.append(0)
                else:
                    pred_outputs.append(1)
            self.preprocessed_data['Prediction'] = pred_outputs
            return self.preprocessed_data
