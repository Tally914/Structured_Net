from utils import *

# Hyper Parameters
lr = .0001  # base lr for cyclic callback
cyclic_lr = CyclicLR(base_lr=lr, max_lr=.001)
optimizer = Nadam(lr=lr)  # possibly use Nadam
loss = rmse

class Structure:
    def __init__(self, dataframe, targets, cat_cols, con_cols, cat_mapper, con_mapper):
        # self.data = shfl(dataframe, random_state=random_state)
        self.dataframe = dataframe
        self.random_state = 42
        self.targets = targets
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.cat_mapper = cat_mapper
        self.con_mapper = con_mapper
        # for col in con_cols:
        #     self.data[col] = self.data[col].replace([np.inf, -np.inf, np.nan, np.NaN, -1.0], np.NaN)
        #     self.data[col].fillna(self.data[col].median(), inplace=True)
        #     self.data[col].replace(np.NaN, 10000000000.0, inplace=True)
        #
        #
        # for col in cat_cols:
        #     self.data[col] = self.data[col].replace([np.inf, -np.inf, np.nan, np.NaN, -1.0], np.NaN)
        #     self.data[col].fillna("None", inplace=True)

    # def __init__(self, database, targets, cat_cols, con_cols, random_state=42):
    #     self.database = database
    #     self.data = shfl(dataframe, random_state=random_state)
    #     self.random_state = random_state
    #     self.targets = targets
    #     self.cat_cols = cat_cols
    #     self.con_cols = con_cols
    #     for col in con_cols:
    #         self.data[col] = self.data[col].replace([np.inf, -np.inf, np.nan, np.NaN, -1.0], np.NaN)
    #         self.data[col].fillna(self.data[col].median(), inplace=True)
    #         self.data[col].replace(np.NaN, 10000000000.0, inplace=True)
    #
    #
    #     for col in cat_cols:
    #         self.data[col] = self.data[col].replace([np.inf, -np.inf, np.nan, np.NaN, -1.0], np.NaN)
    #         self.data[col].fillna("None", inplace=True)

    def processing(self):

        cat_cols = self.cat_cols
        con_cols = self.con_cols

        cat_maps = [(o, LabelEncoder()) for o in cat_cols]
        con_maps = [([o], StandardScaler()) for o in con_cols]

        cat_mapper = DataFrameMapper(cat_maps)
        cat_map_fit = cat_mapper.fit(self.data)

        contin_mapper = DataFrameMapper(con_maps)
        contin_map_fit = contin_mapper.fit(self.data)

        self.cat_map_fit = cat_map_fit
        self.contin_map_fit = contin_map_fit

    def emb_process(self, data):
        cat_map = cat_preproc(data, self.cat_mapper)
        con_map = con_preproc(data, self.con_mapper)

        features = split_cols(cat_map) + split_cols(con_map)
        targets = data[self.targets]

        return features, targets

    def gen_wrapper(self, generator):
        data = next(generator)
        while True:
            yield self.emb_process(data[self.cat_cols+self.con_cols]), data[self.targets]

    def raw_gen(self):
        while True:
            data = next(self.dataframe)
            cats, cons, tars = data[self.cat_cols], data[self.con_cols], data[self.targets]
            yield self.dataset_proc(cats, cons, tars)

    def dataset_proc(self, cats, cons, tars):
        cat_map = cat_preproc(cats, self.cat_mapper)
        con_map = con_preproc(cons, self.con_mapper)

        features = split_cols(cat_map) + split_cols(con_map)
        return features, tars

    def gen_dataset(self, generator, batch_size):
        dataset = Dataset.from_generator(generator, (tf.int32, tf.float32, tf.float32),
                                         (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None])))

        def wrapped_calc(cats, cons, tars):
            return self.dataset_proc(cats, cons, tars)


        dataset.shuffle(batch_size)
        dataset = dataset.map(wrapped_calc, num_parallel_calls=6)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)

        return dataset


    def input_fn(self, batch_size):

        col_defaults = [[0] for i in range(len(self.cat_cols))] + \
                       [[0.0] for i in range(len(self.con_cols))] + \
                       [[0.0] for i in range(len(self.targets))]



        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=col_defaults)

            inputs = {}
            labels = None

            features = dict(zip(self.cat_cols+self.con_cols + self.targets, columns))
            features = columns[:len(self.cat_cols+self.con_cols)]
            targets = columns[len(self.cat_cols+self.con_cols):]

            for key in features:
                if key in self.cat_cols + self.con_cols:
                    inputs[key] = features[key]
                elif key in self.targets:
                    labels = features[key]

            return inputs, labels

        # dataset = tf.data.Dataset.from_generator()
        dataset.shuffle(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=6)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        print('Dataset created')
        return dataset

    def dataset_gen(self, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
        while True:
            yield next_batch

    # def img_process(self, data):
    #     def norm_pixels(array):
    #         return plot_array(array, log=False, resize=(150, 150), plot=False)
    #
    #     dfs = [data[self.ts_cols[i]] for i in range(len(self.ts_cols))]
    #     norms = [np.apply_along_axis(func1d=norm_pixels, arr=dfs[i], axis=1) for i in range(len(dfs))]
    #     return norms

    def embedding_train(self, epochs, batch_size, lr, validation_split, layers=[], dropouts=[], model='new',
                        shuffle=False, save=False, con_dim=10):

        # dataset = self.gen_dataset(self.raw_gen, batch_size)
        # gen = self.dataset_gen(dataset)
        gen= self.raw_gen()
        # trn_gen, val_gen = trn_val_gens(self.data, self.data[self.targets], batch_size, validation_split,
        #                                 preprocessing=self.emb_process)

        # cyclic_lr = CyclicLR(base_lr=lr, max_lr=lr * 100, step_size=(len(trn_gen) + len(val_gen)) / batch_size * 2,
        #                      gamma=0.99)

        cyclic_lr = CyclicLR(base_lr=lr, max_lr=lr * 100, step_size=1000, gamma=0.99)

        checkpoints = ModelCheckpoint('emb_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        emb_outs = [cat_map_info(feat)[1] for feat in self.cat_mapper.features]
        cat_outs = [(i + 1) // 2 if (i + 1) // 2 <= 50 else 50 for i in emb_outs]
        con_outs = [con_dim for i in range(len(self.con_cols))]

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)
        if model == 'new':
            embs = [struc_emb(feat) for feat in self.cat_mapper.features]
            conts = [struc_con(feat, con_dim) for feat in self.con_mapper.features]
            # conts =  struc_costrk.cat_cols+strk.con_colsns(self.contin_map_fit.features, con_dim)

            x = concatenate([e for inp, e in embs] + [d for inp, d in conts], name='concat_outputs')
            # x = concatenate([e for inp,e in embs] + [conts[0]], name='concat_outputs')

            for i in range(len(layers)):
                x = Dense(layers[i], activation='elu')(x)
                x = Dropout(dropouts[i])(x)

            loss = rmse
            output = Dense(1, activation='linear')(x)
            model = Model(inputs=[inp for inp, e in embs] + [inp for inp, d in conts], outputs=output)
            model.compile(optimizer=Nadam(lr=lr), loss=[loss])

            # model = Model(inputs=[inp for inp, e in embs] + [conts[0]], outputs=output)


        # history = model.fit_generator(trn_gen, epochs=epochs,
        #                               verbose=1, validation_data=val_gen, callbacks=callbacks, shuffle=shuffle,
        #                               workers=multiprocessing.cpu_count())

        history = model.fit_generator(gen, epochs=epochs, steps_per_epoch= 1000,
                                      verbose=1, callbacks=callbacks, shuffle=shuffle,
                                      workers=multiprocessing.cpu_count())

        self.emb_model = model
        self.emb_history = history
        self.cat_dims = cat_outs
        self.con_dims = con_outs

    def chart_train(self, epochs, batch_size, lr, validation_split, layers=[], dropouts=[], model='new',
                    shuffle=False, save=False):
        trn_gen, val_gen = trn_val_gens(self.data, self.data[self.targets], batch_size, validation_split,
                                        preprocessing=self.img_process)

        cyclic_lr = CyclicLR(base_lr=lr, max_lr=lr * 100, step_size=len(trn_gen) / batch_size * 2, gamma=0.99)
        checkpoints = ModelCheckpoint('img_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)

        if model == 'new':
            inception = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3),
                                          pooling='avg')

            inp_list = [Input(shape=(150, 150, 3)) for i in range(len(self.ts_cols))]
            out_list = [inception(inp) for inp in inp_list]
            # dense_outs = [Dense(layers[0], activation='elu')(out) for out in out_list]

            x = concatenate(out_list, name='concat_dense_outputs')
            for i in range(len(layers)):
                x = Dense(layers[i], activation='elu', name=f'img_dense_{i}')(x)
                x = Dropout(dropouts[i], name=f'img_dropout_{i}')(x)

            output = Dense(1, activation='linear')(x)
            for layer in inception.layers: layer.trainable = False

            model = Model(input=inp_list, output=output)

            model.compile(optimizer=Nadam(lr=lr), loss=[loss])

        history = model.fit_generator(trn_gen, steps_per_epoch= 1000, epochs=epochs, verbose=1, validation_data=val_gen, callbacks=callbacks,
                                      shuffle=shuffle, workers=multiprocessing.cpu_count())

        self.img_model = model
        self.img_history = history

    def intermediate_outputs(self):
        emb_out = layer_out_raw(self.emb_process(self.data), self.emb_model, 'concat_outputs')
        img_out = layer_out_raw(self.img_process(self.data), self.img_model, 'concat_dense_outputs')


        intermediate = pd.concat([emb_out, img_out], axis=1)
        intermediate.to_csv('C:\\Users\\tales\Code\crypto_process\models\intermediate_outputs.csv')

    def load_emb_model(self, path):
        print('Loading emb_model...')
        self.emb_model = load_model(path, custom_objects={'rmse': rmse})
        print('Loaded emb_model!')

    def load_img_model(self, path):
        print('Loading img_model...')
        self.img_model = load_model(path, custom_objects={'rmse': rmse})
        print('Loaded img_model!')

    def load_nn_model(self, path):
        print('Loading nn_model...')
        self.nn_model = load_model(path, custom_objects={'rmse': rmse})
        print('Loaded nn_model!')

    def load_intermediates(self, path):
        self.intermediate = pd.read_csv(path)

    def nn_train(self, epochs, batch_size, lr, validation_split, layers=[], dropouts=[], model='new', shuffle=False,
                 save=False):

        trn_gen, val_gen = trn_val_gens(self.intermediate, self.data[self.targets], batch_size, validation_split,
                                        preprocessing=None)

        cyclic_lr = CyclicLR(base_lr=lr, max_lr=lr * 100, step_size=len(self.intermediate) / batch_size * 2, gamma=0.99)
        checkpoints = ModelCheckpoint('nn_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)
        input = Input(shape=(self.intermediate.shape[1],))
        x = Dense(layers[0], activation='elu')(input)
        if model == 'new':

            for i in range(len(layers)):
                x = Dense(layers[i], activation='elu')(x)
                x = Dropout(dropouts[i])(x)

            output = Dense(1, activation='linear')(x)
            model = Model(inputs=input, outputs=output)

            model.compile(optimizer=optimizer, loss=[rmse])

        history = model.fit_generator(trn_gen, epochs=epochs, verbose=1, validation_data=val_gen, callbacks=callbacks,
                                      shuffle=shuffle, workers=multiprocessing.cpu_count())


        self.nn_model = model
        self.nn_history = history


class Estimator:
    def __init__(self, train_data, test_data, feature_columns, target_columns):
        self.train_data = {}
        for key in list(train_data[feature_columns]):
            vals = train_data[key].values
            self.train_data[key] = vals.astype(float)
        self.train_targets = train_data[target_columns].values.astype(float)
        print('Train data processed!')

        self.test_data = {}
        for key in list(test_data[feature_columns]):
            vals = test_data[key].values
            self.test_data[key] = vals.astype(float)
        self.test_targets = test_data[target_columns].values.astype(float)
        print('Test data processed!')

        self.bucketized_columns = []
        for col in feature_columns:
            data = train_data[col].astype(float)
            boundaries = sorted([data.min(), data.mean(), data.max()])
            numeric_col = tf.feature_column.numeric_column(col)
            try:
                bucketized_col = tf.feature_column.bucketized_column(numeric_col, boundaries=boundaries)
            except ValueError:
                bucketized_col = tf.feature_column.bucketized_column(numeric_col, boundaries=[-1.0, 0.0, 1.0])
            self.bucketized_columns.append(bucketized_col)
        self.feature_columns = feature_columns
        self.target_columns = target_columns

    def train_input_fn(self, feature_dict, label_array, batch_size):
        """An input function for training"""

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict, label_array))

        # Shuffle, repeat, and batch the examples.
        return dataset.shuffle(42).repeat().batch(batch_size)

    def boosted_tree(self, type, batch_size):
        if type == 'regressor':
            tree = estimator.BoostedTreesRegressor(self.bucketized_columns,
                                                   n_batches_per_layer=int(len(self.train_data) / batch_size))
        elif type == 'classifier':
            n_classes = len(np.unique(self.test_targets))
            print(n_classes)
            tree = estimator.BoostedTreesClassifier(self.bucketized_columns,
                                                    n_batches_per_layer=int(len(self.train_data) / batch_size),
                                                    n_classes=len(np.unique(self.test_targets)))
        else:
            print('Error: Incorrect tree_type passed. Breaking')
            tree = "None"

        tree.train(input_fn=lambda: self.train_input_fn(self.train_data, self.train_targets, batch_size))

        return tree

    def dnn(self, type, batch_size):
        if type == 'regressor':

            reg = estimator.DNNLinearCombinedRegressor(dnn_feature_columns=self.bucketized_columns,
                                                       dnn_activation_fn=tf.nn.elu,
                                                       dnn_hidden_units=[100, 100, 100])
        elif type == 'classifier':
            reg = estimator.DNNLinearCombinedClassifier(dnn_feature_columns=self.bucketized_columns,
                                                        dnn_activation_fn=tf.nn.elu,
                                                        dnn_hidden_units=[100, 100, 100],
                                                        n_classes=len(np.unique(self.test_targets)))
        else:
            print('Error: Incorrect net_type passed. Breaking')
            reg = "None"

        reg.train(input_fn=lambda: self.train_input_fn(self.train_data, self.train_targets, batch_size))
        return reg

batch_size = 256

complete_data = pd.read_csv("Mapped_Augmented_Train.csv", chunksize=batch_size)

sample = next(complete_data)

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

cat_mapper = unpickle("C:\\Users\\tales\Code\structured_analysis\cat_mapper.pkl")
con_mapper = unpickle("C:\\Users\\tales\Code\structured_analysis\con_mapper.pkl")

data = pd.read_csv("Augmented_Train.csv", chunksize=batch_size)

strk = Structure(data, target_col, cat_cols, con_cols, cat_mapper, con_mapper)

strk.embedding_train(3, batch_size, 0.001, 0.1, layers=[256, 256, 256], dropouts=[0.5, 0.5, 0.5], model='new',
                        shuffle=False, save=False, con_dim=10)