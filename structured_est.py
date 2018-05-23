import tensorflow as tf
from tensorflow.python.keras import Model, Input, Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout

from tensorflow.python.keras.callbacks import *
from tensorflow.python.data import Dataset, Iterator
from tensorflow.python import feature_column as fc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python import estimator as est
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn_pandas import DataFrameMapper

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

pandas_csv = pd.read_csv("Mapped_Augmented_Train.csv")

sample = pandas_csv

tar_periods = 7

target_cat_cols = [f'{i+1}_per_tar_cat' for i in range(tar_periods)]

target_val_cols = [f'{i+1}_per_tar_val' for i in range(tar_periods)]

target_cols = target_cat_cols+target_val_cols

target_col = ['3_per_tar_val']

drop_cols = [
 'Unnamed: 0',
 'index',
 'id'
]

cat_cols = [
 'name',
 'symbol',
 'rank',
 'max_sell_ex',
 'min_sell_ex',
 'max_buy_ex',
 'min_buy_ex',
 'Event_Count',
 'Time',
 'Time_Year',
 'Time_Month',
 'Time_Week',
 'Time_Day',
 'Time_Dayofweek',
 'Time_Dayofyear',
 'Time_Hour',
 'Time_Minute',
 'Time_Is_month_end',
 'Time_Is_month_start',
 'Time_Is_quarter_end',
 'Time_Is_quarter_start',
 'Time_Is_year_end',
 'Time_Is_year_start',
 'Time_Elapsed'
]

con_cols = [col for col in list(sample) if col not in target_cols+drop_cols+cat_cols]


class Structured:
    def __init__(self, data_file, cat_cols, con_cols, target_cat_cols, target_con_cols, target_cols, model_dir='None'):
        """
        Inputs
            data_file: csv file that data will be pulled from
            cat_cols: list of categorical columns mapped to integers
            con_cols: list of continuous columns mapped to floats
            target_cat_cols: list of possible target columns mapped to floats
            target_cat_cols: list of possible target columns mapped to ints
            target_cols: list of actual columns to be predicted
        """

        self.data_file = data_file
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.target_cat_cols = target_cat_cols
        self.target_con_cols = target_con_cols
        self.target_cols = target_col
        self.model_dir = model_dir
        self.cat_fc = [
            fc.embedding_column(fc.categorical_column_with_identity(key=col, num_buckets=100, default_value=0),
                                sample[col].nunique()) for col in self.cat_cols]
        self.con_fc = [fc.numeric_column(key=col, default_value=0) for col in self.con_cols]

    def input_fn(self, data_file, batch_size, epochs, feature_names):

        col_defaults = [[0] for i in range(len(self.cat_cols))] + [[0.0] for i in range(len(self.con_cols))] + [[0.0] for i in range(len(self.target_cat_cols + self.target_con_cols))]

        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=col_defaults)

            inputs = {}
            labels = None

            features = dict(zip(feature_names + self.target_cols, columns))
            for key in features:
                if key in self.cat_cols + self.con_cols:
                    inputs[key] = features[key]
                elif key in self.target_cols:
                    labels = features[key]

            return inputs, labels

        dataset = tf.data.TextLineDataset(data_file).skip(1)
        dataset.shuffle(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=6)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        print('Dataset created')
        return dataset

    def dataset_gen(self, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            yield next_batch

    def my_dnn_regression_fn(self, features, labels, mode, params):
        """A model function implementing DNN regression for a custom Estimator."""

        # Extract the input into a dense layer, according to the feature_columns.
        int_outs = []

        net = tf.feature_column.input_layer(features, params["feature_columns"])

        # Iterate over the "hidden_units" list of layer sizes, default is [20].
        units = params.get("hidden_units", [100])
        dropouts = params.get("dropouts", [0.5])
        for i in range(len(units)):
            net = tf.layers.dense(inputs=net, units=units[i], activation=tf.nn.elu, name=f"dense_layer_{i}")
            int_outs.append(net.copy())
            net = tf.layers.batch_normalization(inputs=net, name=f"bn_layer_{i}")
            net = tf.layers.dropout(inputs=net, rate=dropouts[i], name=f"dropout_layer_{i}")

        # Connect a linear output layer on top.
        output_layer = tf.layers.dense(inputs=net, units=1)

        # Reshape the output layer to a 1-dim Tensor to return predictions
        predictions = tf.squeeze(output_layer, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:

            pred_layer = params.get("intermediate_preds", None)
            # In `PREDICT` mode we only need to return predictions.
            if pred_layer == None:
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions={self.target_cols[0]: predictions})
            else:
                int_preds = int_outs[pred_layer]
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions=int_preds)

            # Calculate loss using mean squared error
        average_loss = tf.losses.mean_squared_error(labels, predictions)

        # Pre-made estimators use the total_loss instead of the average,
        # so report total_loss for compatibility.
        batch_size = tf.shape(labels)[0]
        total_loss = tf.to_float(batch_size) * average_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params.get("optimizer", tf.train.AdamOptimizer)
            optimizer = optimizer(params.get("learning_rate", 0.001))
            train_op = optimizer.minimize(
                loss=average_loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)
        # In evaluation mode we will calculate evaluation metrics.
        assert mode == tf.estimator.ModeKeys.EVAL

        # Calculate root mean squared error
        print(labels)
        print(predictions)

        # Fixed for #4083
        predictions = tf.cast(predictions, tf.float64)

        rmse = tf.metrics.root_mean_squared_error(labels, predictions)

        # Add the rmse to the collection of evaluation metrics.
        eval_metrics = {"rmse": rmse}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            # Report sum of error for compatibility with pre-made estimators
            loss=total_loss,
            eval_metric_ops=eval_metrics)

    def ds_to_iter(self, batch_size):
        dataset = self.input_fn(batch_size, self.cat_fc+self.con_fc)
        return dataset.make_one_shot_iterator().get_next()

    def custom_est(self, units, dropouts, batch_size, optimizer=tf.train.AdamOptimizer, learning_rate=0.001):
        feature_columns = self.cat_fc+self.con_fc

        # train_input_fn = self.ds_to_iter(batch_size)

        model = tf.estimator.Estimator(
            model_fn=self.my_dnn_regression_fn,
            params={
                "feature_columns": feature_columns,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "hidden_units": units,
                "dropouts": dropouts
            }
        )

        return model

    def custom_net(self, units, dropout, batch_size, epochs):


        model = Sequential()
        model.add(Dense(units[0], activation="elu", input_shape=(len(cat_cols + con_cols),)))
        for i in range(1, len(units)):
            model.add(Dense(units[i], activation="elu"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='mse')
        # self.nn_inp_names = model.input_names
        return model

    def est_build(self, units, dropout, target_cols):
        if self.model_dir != 'None':
            warm_start = self.model_dir
        else:
            warm_start = None

        estimator = est.DNNRegressor(
            hidden_units=units,
            feature_columns=self.cat_fc + self.con_fc,
            model_dir=self.model_dir,
            label_dimension=len(target_cols),
            weight_column=None,
            optimizer=tf.train.AdamOptimizer(),
            activation_fn=tf.nn.elu,
            dropout=dropout,
            input_layer_partitioner=None,
            config=None,
            warm_start_from=warm_start,
        )
        return estimator

    def est_train(self, estimator, data_file, batch_size, epochs, features):
        def train_input_fn():
            return self.input_fn(data_file, batch_size, epochs, features)

        estimator.train(input_fn=train_input_fn, steps=None)

    def est_evaluate(self, estimator, data_file, batch_size, epochs, features):
        def eval_input_fn():
            return self.input_fn(data_file, batch_size, epochs, features)

        metrics = estimator.evaluate(input_fn=eval_input_fn, steps=None)
        return metrics

    def est_test(self, estimator, data_file, batch_size, epochs, features):
        def test_input_fn():
            return self.input_fn(data_file, batch_size, epochs, features)

        predictions = estimator.predict(input_fn=test_input_fn, steps=None)
        return predictions

    
