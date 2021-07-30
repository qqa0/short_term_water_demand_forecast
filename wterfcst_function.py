import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.layers import Dense,LSTM,GRU,Conv1D,Input,concatenate,add,Lambda,MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class Wterfcst_function:

    @classmethod
    def measure(cls,actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        a = np.sum((actual - predicted) ** 2)
        b = np.sum((actual - np.mean(actual)) ** 2)
        nse = 1 - a / b

        return rmse, mae, mape, nse


    @classmethod
    def select(cls,tra_l, data, seed_num):
        np.random.seed(seed_num)
        index = np.random.randint(0, len(data) / 96 - tra_l - 6)
        sel_data = data[index * 96:(index + tra_l + 6) * 96, :]

        return sel_data, index

    @classmethod
    def to_supervised(cls,data, w, h):
        n, m = data.shape
        s = w + h - 1
        X = np.zeros((n - s, w, m))
        Y = np.zeros((n - s, m))
        for i in range(s, n):
            X[i - s] = data[i - h + 1 - w:i - h + 1].copy()
            Y[i - s] = data[i].copy()
        Y = Y[:, -1].reshape(-1, 1)

        return X, Y

    @classmethod
    def to_split(cls,data):
        n = len(data) / 96
        tn = int((n - 6) * 96)
        vd = int((n - 3) * 96)

        if len(data.shape) == 3:
            tra = data[:tn, :, :].copy()
            val = data[tn:vd, :, :].copy()
            tes = data[vd:, :, :].copy()

        if len(data.shape) == 2:
            tra = data[:tn, :].copy()
            val = data[tn:vd, :].copy()
            tes = data[vd:, :].copy()
        return tra, val, tes

    @classmethod
    def grun_model(cls):
        xdinput = Input(shape=(5, 1))
        xd = GRU(48, return_sequences=True)(xdinput)
        xd = GRU(32, return_sequences=True)(xd)
        xd = GRU(32)(xd)

        xninput = Input(shape=(5, 1))
        xn = GRU(48, return_sequences=True)(xninput)
        xn = GRU(32, return_sequences=True)(xn)
        xn = GRU(32)(xn)

        xrinput = Input(shape=(5, 1))
        xr = GRU(48, return_sequences=True)(xrinput)
        xr = GRU(32, return_sequences=True)(xr)
        xr = GRU(32)(xr)

        x = concatenate([xd, xn, xr])

        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(4, activation='relu')(x)
        x = Dense(2, activation='relu')(x)
        xo = Dense(1)(x)

        model = Model(inputs=[xdinput, xninput, xrinput], outputs=xo)

        return model

    @classmethod
    def grun_fit(cls,model, grun_tra_x, grun_tra_y, grun_val_x, grun_val_y, seed_num, time_length,
                 file_name):

        adam = Adam(lr=0.002)
        model.compile(loss='mae', optimizer=adam)
        filepath = file_name + '/' + 'grun_weights_' + str(seed_num) + '_' + str(time_length) + '.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                                save_best_only=False,
                                                                mode='min')
        # fit
        model.fit(grun_tra_x, [grun_tra_y],
                  validation_data=(grun_val_x, [grun_val_y]),
                  epochs=25, batch_size=128, verbose=1,
                  callbacks=[checkpoint], shuffle=True)

    @classmethod
    def conv_gru_model(cls):
        x = Input(shape=(96 * 2, 1))

        x1 = Lambda(lambda k: k[:, :96, :])(x)
        x1 = Conv1D(64, 1, padding="same", activation='relu')(x1)
        x11 = Conv1D(64, 12, padding="same", activation='relu')(x1)
        x12 = Conv1D(64, 24, padding="same", activation='relu')(x1)
        x111 = Conv1D(32, 1, padding="same", activation='relu')(x11)
        x112 = Conv1D(16, 1, padding="same", activation='relu')(x11)
        x113 = Conv1D(8, 1, padding="same", activation='relu')(x11)
        x11 = concatenate([x111, x112, x113])
        x11 = GRU(48, return_sequences=True)(x11)
        x11 = GRU(32)(x11)
        x121 = Conv1D(32, 1, padding="same", activation='relu')(x12)
        x122 = Conv1D(16, 1, padding="same", activation='relu')(x12)
        x123 = Conv1D(8, 1, padding="same", activation='relu')(x12)
        x12 = concatenate([x121, x122, x123])
        x12 = GRU(48, return_sequences=True)(x12)
        x12 = GRU(32)(x12)

        x2 = Lambda(lambda k: k[:, 96:, :])(x)
        x2 = Conv1D(64, 1, padding="same", activation='relu')(x2)
        x21 = Conv1D(64, 12, padding="same", activation='relu')(x2)
        x22 = Conv1D(64, 24, padding="same", activation='relu')(x2)
        x211 = Conv1D(32, 1, padding="same", activation='relu')(x21)
        x212 = Conv1D(16, 1, padding="same", activation='relu')(x21)
        x213 = Conv1D(8, 1, padding="same", activation='relu')(x21)
        x21 = concatenate([x211, x212, x213])
        x21 = GRU(48, return_sequences=True)(x21)
        x21 = GRU(32)(x21)

        x221 = Conv1D(32, 1, padding="same", activation='relu')(x22)
        x222 = Conv1D(16, 1, padding="same", activation='relu')(x22)
        x223 = Conv1D(8, 1, padding="same", activation='relu')(x22)
        x22 = concatenate([x221, x222, x223])
        x22 = GRU(48, return_sequences=True)(x22)
        x22 = GRU(32)(x22)

        x1_2 = concatenate([x11, x12, x21, x22])
        x1_2 = Dense(128, activation='relu')(x1_2)

        x3 = Conv1D(64, 1, padding="same", activation='relu')(x)
        x3 = GRU(48, return_sequences=True)(x3)
        x3 = GRU(32)(x3)
        x3 = Dense(128, activation='relu')(x3)
        x_m = concatenate([x1_2, x3])

        xo = Dense(128, activation='relu')(x_m)
        xo = Dense(64, activation='relu')(xo)
        xo = Dense(32, activation='relu')(xo)
        xo = Dense(16, activation='relu')(xo)
        xo = Dense(8, activation='relu')(xo)
        xo = Dense(4, activation='relu')(xo)
        xo = Dense(1)(xo)
        model = Model(inputs=[x], outputs=xo)

        return model

    @classmethod
    def conv_gru_fit(cls,model, tra_x, tra_y, val_x, val_y, seed_num, time_length, file_name):

        adam = Adam(lr=0.002)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                                  patience=15, verbose=0, mode='min')
        filepath = file_name + '/' + 'conv_gru_weights_' + str(seed_num) + '_' + str(time_length) + '.best.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                                save_best_only=True,
                                                                mode='min')
        model.compile(loss='mae', optimizer=adam)

        # fit
        model.fit([tra_x], [tra_y],
                  validation_data=([val_x], [val_y]),
                  epochs=1000, batch_size=128, verbose=1,
                  callbacks=[early_stopping, checkpoint], shuffle=True)

    @classmethod
    def ann_model(cls):
        xinput = Input(shape=(192, 1))
        x = Lambda(lambda k: K.reshape(k, (-1, 192)))(xinput)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(4, activation='relu')(x)
        x = Dense(1)(x)
        model = Model(inputs=[xinput], outputs=[x])
        return model

    @classmethod
    def ann_fit(cls,model, tra_x, tra_y, val_x, val_y, seed_num, time_length, file_name):

        adam = Adam(lr=0.002)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=20, verbose=0, mode='min')
        filepath = file_name + '/' + 'ann_weights_' + str(seed_num) + '_' + str(time_length) + '.best.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                     mode='min')
        model.compile(loss='mae', optimizer=adam)

        # fit
        model.fit([tra_x], [tra_y],
                  validation_data=([val_x], [val_y]),
                  epochs=1000, batch_size=60, verbose=1,
                  callbacks=[early_stopping, checkpoint], shuffle=True)

    @classmethod
    def trans_back(cls,data, m):
        X = np.zeros((len(data), m))
        data = data.reshape(-1)
        X[:, -1] = data

        return X





