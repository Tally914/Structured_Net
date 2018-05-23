import tensorflow as tf
from tensorflow.python.keras import Model, Input, Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from utils import run_tensorboard
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.utils import to_categorical
from google.datalab.ml import TensorBoard
from tensorflow.python.data import Dataset, Iterator
from tensorflow.python import feature_column as fc
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python import estimator as est
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.python import debug as tf_debug
from sklearn_pandas import DataFrameMapper

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

pandas_csv = pd.read_csv("Mapped_Augmented_Train.csv")

sample = pandas_csv

tar_periods = 7

target_cat_cols = [f'{i+1}_per_tar_cat' for i in range(tar_periods)]

target_val_cols = [f'{i+1}_per_tar_val' for i in range(tar_periods)]

target_cols = target_cat_cols + target_val_cols

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

con_cols = [col for col in list(sample) if col not in target_cols + drop_cols + cat_cols]
debug_port = 9001

class Structured:
    def __init__(self, data_file, cat_cols, con_cols, target_cat_cols, target_con_cols, target_cols, model_dir='None', debug_port=9001):
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
        self.features = cat_cols+con_cols
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.target_cat_cols = target_cat_cols
        self.target_con_cols = target_con_cols
        self.target_cols = target_cols
        self.model_dir = model_dir
        self.cat_fc = [fc.embedding_column(fc.categorical_column_with_identity(key=col, num_buckets=100, default_value=0),
                                sample[col].nunique()) for col in self.cat_cols]
        self.con_fc = [fc.numeric_column(key=col, default_value=0) for col in self.con_cols]
        self.debug_port = debug_port
        # run_tensorboard(model_dir, debug_port)

    def my_dnn_regression_fn(self, features, labels, mode, params):

        # debug_hook = tf_debug.TensorBoardDebugHook(f"DESKTOP-8KR8FQT:{self.debug_port}")

        preds = {}

        inps = tf.feature_column.input_layer(features, params["feature_columns"])
        preds["inputs"] = inps

        embs = tf.layers.dense(inputs=inps, units=params["embedding_dims"], activation=tf.nn.elu, name=f"embedding_layer")
        preds[f"embeddings"] = embs

        net = tf.layers.batch_normalization(inputs=embs, name=f"inp_norm_layer")
        preds["inp_norm_layer"] = net

        units = params.get("hidden_units", [100])
        dropouts = params.get("dropouts", [0.5])
        for i in range(len(units)):
            net = tf.layers.dense(inputs=net, units=units[i], activation=tf.nn.elu, name=f"dense_layer_{i}")
            preds[f"dense_layer_{i}"] = net

            net = tf.layers.batch_normalization(inputs=net, name=f"bn_layer_{i}")
            preds[f"bn_layer_{i}"] = net

            net = tf.layers.dropout(inputs=net, rate=dropouts[i], name=f"dropout_layer_{i}")
            preds[f"dropout_layer_{i}"] = net


        output_layer = tf.layers.dense(inputs=net, units=1)

        predictions = tf.squeeze(output_layer, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:

            preds[self.target_cols[0]] = predictions
            return tf.estimator.EstimatorSpec(mode=mode, predictions=preds)


        average_loss = tf.losses.mean_squared_error(labels, predictions)

        batch_size = tf.shape(labels)[0]
        total_loss = tf.to_float(batch_size) * average_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params.get("optimizer", tf.train.AdamOptimizer)
            optimizer = optimizer(params.get("learning_rate", 0.001))
            train_op = optimizer.minimize(
                loss=average_loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        assert mode == tf.estimator.ModeKeys.EVAL

        print(labels)
        print(predictions)

        mse = tf.metrics.mean_squared_error(labels, predictions)

        rmse = tf.metrics.root_mean_squared_error(labels, predictions)

        eval_metrics = {'mse': mse, 'rmse': rmse}


        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics)

    def input_fn(self, data_file, num_epochs, batch_size, feature_names):

        col_defaults = [[0] for i in range(len(self.cat_cols))]+[[0.0] for i in range(len(self.con_cols))]+[[0.0] for i in range(len(self.target_cat_cols+self.target_con_cols))]


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
        # dataset = dataset.shard(multiprocessing.cpu_count() , FLAGS.worker_index)

        dataset.shuffle(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=4)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        print('Dataset created')
        return dataset

    def gen_input_fn(self, data_file, num_epochs, batch_size, feature_names):

        col_defaults = [[0.0] for i in range(len(feature_names)+1)]

        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=col_defaults)
            inputs = {}
            labels = None

            features = dict(zip(feature_names + self.target_cols, columns))
            for key in features:
                if key in feature_names:

                    inputs[key] = features[key]
                elif key in self.target_cols:
                    labels = features[key]

            return inputs, labels

        dataset = tf.data.TextLineDataset(data_file).skip(1)

        dataset.shuffle(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=4)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        print('Dataset created')
        return dataset

    def custom_est(self, units, dropouts, embedding_dims, optimizer=tf.train.AdamOptimizer, learning_rate=0.001, warm_start_from=None):
        feature_columns = self.cat_fc + self.con_fc


        model = tf.estimator.Estimator(
            model_fn=self.my_dnn_regression_fn,
            model_dir=self.model_dir,
            warm_start_from=warm_start_from,
            params={
                "feature_columns": feature_columns,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "hidden_units": units,
                "dropouts": dropouts,
                "embedding_dims": embedding_dims
            }
        )
        return model

    def gbt_est(self, feature_cols):
        boundaries = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0]
        feature_cols = [fc.bucketized_column(fc.numeric_column(key=col, default_value=0), boundaries=boundaries) for col in feature_cols]
        estimator = est.BoostedTreesRegressor(feature_cols, n_batches_per_layer=100)
        return estimator

    def est_train(self, estimator, data_file, epochs, batch_size, features):
        def train_input_fn():
            return self.input_fn(data_file, epochs, batch_size, features)


        estimator.train(input_fn=train_input_fn, steps=10000)

    def est_evaluate(self, estimator, data_file, batch_size, features):
        def eval_input_fn():
            return self.input_fn(data_file, 1, batch_size, features)

        metrics = estimator.evaluate(input_fn=eval_input_fn, steps=300)
        return metrics

    def est_predict(self, estimator, data_file, batch_size, features):
        def test_input_fn():
            return self.input_fn(data_file, 1, batch_size, features)

        pred_gen = estimator.predict(input_fn=test_input_fn)
        # steps=1
        # predictions = []
        # for pred in pred_gen:
        #     steps += 1
        #     if steps % 100 == 0:
        #         print(f'Prediction batch: {steps}.')
        #     predictions.append(pred)
        # return predictions
        batch_1 = next(pred_gen)
        self.inp_shape = batch_1["embeddings"].shape[0]
        pred_gen= estimator.predict(input_fn=test_input_fn)

        # emb_inp_shape = batch_1["inputs"].shape[0]
        return pred_gen

    def gbt_est_predict(self, estimator, data_file, num_epochs, batch_size, features):
        def test_input_fn():
            return strk.gen_input_fn(data_file, num_epochs, batch_size, features)

        estimator.train(input_fn=test_input_fn)


    def save_preds(self, pred_gen, csv_path, features, targets):
        count = 1
        csv_list= []
        for pred in pred_gen:
            embs = pred[features].tolist()
            target = [pred[targets].tolist()]

            csv_list.append(embs+target)
            if count % 1000 == 0:
                df = pd.DataFrame(csv_list, columns=[f"Latent_Factor_{i}" for i in range(len(embs))]+self.target_cols)
                df.to_csv(csv_path, index=None, mode='a')
                csv_list = []
                print(f"Predictions written to csv: {count}.")
            count += 1
        leftovers = pd.DataFrame(csv_list, columns=[f"Latent_Factor_{i}" for i in range(len(embs))]+self.target_cols)
        leftovers.to_csv(csv_path, index=None, mode='a')
        print(f"Predictions written to csv: {count}.")


batch_size = 256
embedding_dims = 300
model_dir = 'C:\\Users\\tales\Code\structured_analysis\strk'

strk = Structured("Mapped_Values.csv",
                  cat_cols,
                  con_cols,
                  target_cat_cols,
                  target_val_cols,
                  target_col,
                  model_dir=model_dir)

model = strk.custom_est(units=[512,512,512,512,512],
                        dropouts=[0.75,0.75,0.50,0.50,0.25],
                        embedding_dims=embedding_dims,
                        optimizer=tf.train.AdamOptimizer,
                        learning_rate=0.0001,
                        warm_start_from=model_dir)


strk.est_train(model, "Mapped_Values.csv", 1, batch_size, strk.features)
strk.est_evaluate(model, "Mapped_Values.csv", batch_size, strk.features)
predictions = strk.est_predict(model, "Mapped_Values.csv", batch_size, strk.features)
# strk.save_preds(predictions, "Intermediates.csv", "embeddings", strk.target_cols[0])


# gbt = strk.gbt_est([f"Latent_Factor_{i}" for i in range(strk.inp_shape)])
# strk.est_train(gbt, "Mapped_Augmented_Train.csv", 1, batch_size, strk.features)

# strk.gbt_est_predict(gbt, "Intermediates.csv", 1, batch_size, [f"Latent_Factor_{i}" for i in range(strk.inp_shape)])


