"""
@author: chenlei

"""
import pandas as pd
from wterfcst_function import *
from sklearn.preprocessing import MinMaxScaler




class Wterfcst:

    def __init__(self,time_length, data, seed_num,file_name):
        self.time_length = time_length
        self.data = data
        self.seed_num = seed_num
        self.file_name = file_name


    def dataset_get(self):
        data, dindex = Wterfcst_function.select(self.time_length, self.data,self.seed_num)

        #GRUN_data
        xd, y = Wterfcst_function.to_supervised(data, 5, 190)
        xn, _ = Wterfcst_function.to_supervised(data, 5, 94)
        xn = xn[len(xn) - len(xd):, :, :]
        xr, _ = Wterfcst_function.to_supervised(data, 5, 1)
        xr = xr[len(xr) - len(xd):, :, :]

        tra_xd, val_xd, tes_xd = Wterfcst_function.to_split(xd)
        tra_xn, val_xn, tes_xn = Wterfcst_function.to_split(xn)
        tra_xr, val_xr, tes_xr = Wterfcst_function.to_split(xr)
        grun_tra_y, grun_val_y, grun_tes_y = Wterfcst_function.to_split(y)

        grun_tra_x = [tra_xd,tra_xn,tra_xr]
        grun_val_x = [val_xd,val_xn,val_xr]
        grun_tes_x = [tes_xd,tes_xn,tes_xr]
        self.grun_dataset = [grun_tra_x,grun_tra_y,grun_val_x,grun_val_y,grun_tes_x,grun_tes_y]

        #ANN and Conv1D-GRU data
        xc, yc = Wterfcst_function.to_supervised(data, 192, 1)
        tra_xc, val_xc, tes_xc = Wterfcst_function.to_split(xc)
        tra_yc, val_yc, tes_yc = Wterfcst_function.to_split(yc)

        self.ac_dataset = [tra_xc,tra_yc,val_xc,val_yc,tes_xc,tes_yc]



    def model_train(self):

        #grun
        grun_model = Wterfcst_function.grun_model()
        Wterfcst_function.grun_fit(grun_model,self.grun_dataset[0],self.grun_dataset[1],
                                   self.grun_dataset[2],self.grun_dataset[3],self.seed_num,
                                   self.time_length,self.file_name)

        #ann
        ann_model = Wterfcst_function.ann_model()
        Wterfcst_function.ann_fit(ann_model,self.ac_dataset[0],self.ac_dataset[1],
                                  self.ac_dataset[2],self.ac_dataset[3],self.seed_num,
                                  self.time_length,self.file_name)

        #conv1d-gru
        conv_gru_model = Wterfcst_function.conv_gru_model()
        Wterfcst_function.conv_gru_fit(conv_gru_model,self.ac_dataset[0],self.ac_dataset[1],
                                  self.ac_dataset[2],self.ac_dataset[3],self.seed_num,
                                  self.time_length,self.file_name)

    def model_pre(self):

        #grun
        grun_model = Wterfcst_function.grun_model()
        grun_weight = self.file_name + '/' + 'grun_weights_' + str(self.seed_num) + '_' + str(self.time_length) + '.hdf5'
        grun_model.load_weights(grun_weight)
        grun_pre_tes = grun_model.predict(self.grun_dataset[4])

        #ann
        ann_model = Wterfcst_function.ann_model()
        ann_weight = self.file_name + '/' + 'ann_weights_' + str(self.seed_num) + '_' + str(self.time_length) + '.best.hdf5'
        ann_model.load_weights(ann_weight)
        ann_pre_tes = ann_model.predict(self.ac_dataset[4])

        #conv1d-gru
        conv_gru_model = Wterfcst_function.conv_gru_model()
        conv_gru_weight = self.file_name + '/' + 'conv_gru_weights_' + str(self.seed_num) + '_' + str(self.time_length) + '.best.hdf5'
        conv_gru_model.load_weights(conv_gru_weight)
        conv_gru_pre_tes = conv_gru_model.predict(self.ac_dataset[4])

        grun_tes_y = Wterfcst_function.trans_back(self.grun_dataset[5],1)
        ac_tes_y = Wterfcst_function.trans_back(self.ac_dataset[5],1)
        grun_pre_tes = Wterfcst_function.trans_back(grun_pre_tes,1)
        ann_pre_tes = Wterfcst_function.trans_back(ann_pre_tes,1)
        conv_gru_pre_tes = Wterfcst_function.trans_back(conv_gru_pre_tes,1)

        result = [grun_tes_y,ac_tes_y,grun_pre_tes,ann_pre_tes,conv_gru_pre_tes]

        return result

    def evaluate(self,result,scaler):
        grun_tes_y_inv = scaler.inverse_transform(result[0])[:,-1]
        ac_tes_y_inv = scaler.inverse_transform(result[1])[:,-1]
        grun_pre_inv = scaler.inverse_transform(result[2])[:,-1]
        ann_pre_inv = scaler.inverse_transform(result[3])[:,-1]
        conv_gru_pre_inv = scaler.inverse_transform(result[4])[:,-1]

        rmse_conv,mae_conv,mape_conv,nse_conv = Wterfcst_function.measure(ac_tes_y_inv,conv_gru_pre_inv)
        rmse_grun,mae_grun,mape_grun,nse_grun = Wterfcst_function.measure(grun_tes_y_inv,grun_pre_inv)
        rmse_ann,mae_ann,mape_ann,nse_ann = Wterfcst_function.measure(ac_tes_y_inv,ann_pre_inv)

        conv_pre = np.array([rmse_conv,mae_conv,mape_conv,nse_conv])
        grun_pre = np.array([rmse_grun,mae_grun,mape_grun,nse_grun])
        ann_pre = np.array([rmse_ann,mae_ann,mape_ann,nse_ann])

        eva_pre = np.array([conv_pre,grun_pre,ann_pre])

        return eva_pre



if __name__ == '__main__':
    df = pd.read_csv('input.csv')
    data = df[['value']].values

    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)

    wter = Wterfcst(60,data,100,'./')
    wter.dataset_get()
    wter.model_train()
    result = wter.model_pre()
    eva_pre = wter.evaluate(result,scaler)


