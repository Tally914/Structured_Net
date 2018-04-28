from utils import *

# Hyper Parameters
lr = .0001  # base lr for cyclic callback
cyclic_lr = CyclicLR(base_lr=lr, max_lr=.001)
optimizer = Nadam(lr=lr)  # possibly use Nadam
loss = rmse


class Structure:
    def __init__(self, dataframe, targets, cat_cols, con_cols, ts_cols, random_state=42):
        self.data = shfl(dataframe, random_state=random_state)
        self.random_state = random_state
        self.targets = targets
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.ts_cols = ts_cols
        self.data[con_cols] = self.data[con_cols].replace([np.inf, -np.inf, np.nan, np.NaN], -1.0)
        self.data[con_cols] = self.data[con_cols].fillna(-1.0)
        self.data[con_cols] = self.data[con_cols].astype(float)
        self.data[cat_cols] = self.data[cat_cols].replace([np.inf, -np.inf, np.nan, np.NaN], "-1.0")
        self.data[cat_cols] = self.data[cat_cols].fillna("-1.0")
        self.data[cat_cols] = self.data[cat_cols].astype(str)
        for list in ts_cols:
            self.data[list] = self.data[list].replace([np.inf, -np.inf, np.nan, np.NaN], -1.0)
            self.data[list] = self.data[list].fillna(-1.0)
            self.data[list] = self.data[list].astype(float)

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
        cat_map = cat_preproc(data, self.cat_map_fit)
        con_map = con_preproc(data, self.contin_map_fit)

        mapped = split_cols(cat_map) + split_cols(con_map)

        return mapped

    def img_process(self, data):
        def norm_pixels(array):
            return plot_array(array, log=False, resize=(150, 150), plot=False)

        dfs = [data[self.ts_cols[i]] for i in range(len(self.ts_cols))]
        norms = [np.apply_along_axis(func1d=norm_pixels, arr=dfs[i], axis=1) for i in range(len(dfs))]
        return norms

    def embedding_train(self, epochs, batch_size, lr, validation_split, layers=[], dropouts=[], model='new',
                        shuffle=False, save=False, con_dim=10):
        trn_gen, val_gen = trn_val_gens(self.data, self.data[self.targets], batch_size, validation_split,
                                        preprocessing=self.emb_process)

        cyclic_lr = CyclicLR(base_lr=lr, max_lr=lr * 100, step_size=(len(trn_gen) + len(val_gen)) / batch_size * 2,
                             gamma=0.99)

        checkpoints = ModelCheckpoint('emb_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        emb_outs = [cat_map_info(feat)[1] for feat in self.cat_map_fit.features]
        cat_outs = [(i + 1) // 2 if (i + 1) // 2 <= 50 else 50 for i in emb_outs]
        con_outs = [con_dim for i in range(len(self.con_cols))]

        callbacks = [cyclic_lr]
        if save == True: callbacks.append(checkpoints)
        if model == 'new':
            embs = [struc_emb(feat) for feat in self.cat_map_fit.features]
            conts = [struc_con(feat, con_dim) for feat in self.contin_map_fit.features]
            # conts =  struc_cons(self.contin_map_fit.features, con_dim)

            x = concatenate([e for inp, e in embs] + [d for inp, d in conts], name='concat_outputs')
            # x = concatenate([e for inp,e in embs] + [conts[0]], name='concat_outputs')

            for i in range(len(layers)):
                x = Dense(layers[i], activation='elu')(x)
                x = Dropout(dropouts[i])(x)

            output = Dense(1, activation='linear')(x)
            model = Model(inputs=[inp for inp, e in embs] + [inp for inp, d in conts], outputs=output)
            # model = Model(inputs=[inp for inp, e in embs] + [conts[0]], outputs=output)

            model.compile(optimizer=Nadam(lr=lr), loss=[rmse])

        history = model.fit_generator(trn_gen, epochs=epochs,
                                      verbose=1, validation_data=val_gen, callbacks=callbacks, shuffle=shuffle,
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

            model.compile(optimizer=Nadam(lr=lr), loss=[rmse])

        history = model.fit_generator(trn_gen, epochs=epochs, verbose=1, validation_data=val_gen, callbacks=callbacks,
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