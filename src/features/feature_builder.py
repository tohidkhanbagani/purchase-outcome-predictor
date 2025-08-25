import pandas as pd
import numpy as np
from scipy import sparse
from typing import Tuple,List,Any,Optional
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_ingestion.data_ingest import Ingest
from src.utils.functions import ModelSolutions






class PredictiveModelData:

    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame
        self.model_solutions = ModelSolutions()
        self.scaler_encoder = StandardScaler()
        self.label_encoder = LabelEncoder()

    def data_preprocessing(self,
                           path:List[str]=['data','processed','preprocessed_data'],
                           file_name:str='preprocessed_data.csv'
                           ):
        df = self.data_frame[[
            'id',
            'category',
            'product',
            'purchase_date',
            'purchase_amount',
            'returned',
            'Preferred_Shopping_Channels']].sort_values(by='id').reset_index().drop('index',axis=1)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])

        df = df.sort_values(['id','purchase_date'])

        # sample_number = min(df['returned'].value_counts().tolist())

        # Step 2: Validation
        validator = Ingest()
        validation_result = validator.data_validate(obj=df)

        if any(v > 0 for v in validation_result['missing_values'].values()):
            df.dropna(inplace=True)

        aggregated_data = df.groupby(['id']).agg(
                                                    purchases=('id','count'),
                                                    returns=('returned',lambda x:  (x=='refund').sum()),
                                                    exchanges=('returned',lambda x: (x=='exchange').sum()),
                                                    kept = ('returned',lambda x: (x=='no').sum())
                                                    ).reset_index()
        
        labels = ['one_time_buyer','infrequent_buyer','frequent_buyer','loyal_buyer']
        bins = [0, 2, 5, 11, float('inf')]

        aggregated_data['segment'] = pd.cut(aggregated_data['purchases'],bins=bins,labels=labels,right=False)

        aggregated_data['return_rate'] = round((aggregated_data['returns']/aggregated_data['purchases'])*100,2)
        aggregated_data['exchange_rate'] = round((aggregated_data['exchanges']/aggregated_data['purchases'])*100,2)
        aggregated_data['kept_rate'] = round((aggregated_data['kept']/aggregated_data['purchases'])*100,2)

        final_df = df.merge(aggregated_data[['id','purchases','return_rate','exchange_rate','kept_rate','segment']],on='id',how='left')

        final_df['days_since_last'] = final_df.groupby('id')['purchase_date'].diff().dt.days.fillna(-1)
        cat_ref = df.groupby('category')['returned'].apply(lambda x: (x=='refund').mean())
        cat_exch = df.groupby('category')['returned'].apply(lambda x: (x=='exchange').mean())
        cat_keep = df.groupby('category')['returned'].apply(lambda x: (x=='no').mean())

        chan_ref = df.groupby('Preferred_Shopping_Channels')['returned'].apply(lambda x: (x=='refund').mean())
        chan_exch = df.groupby('Preferred_Shopping_Channels')['returned'].apply(lambda x: (x=='exchange').mean())
        chan_keep = df.groupby('Preferred_Shopping_Channels')['returned'].apply(lambda x: (x=='no').mean())

        amt_bins = [0,10,50,200,500,1000,float('inf')]
        amt_bin_labels = ['very_low','low','mid','high','very_high','VIP']
        final_df['amt_bin'] =pd.cut(final_df['purchase_amount'],bins=amt_bins,labels=amt_bin_labels)
        final_df['cat_refund_rate'] = final_df['category'].map(cat_ref)
        final_df['cat_exchange_rate'] = final_df['category'].map(cat_exch)
        final_df['cat_kept_rate'] = final_df['category'].map(cat_keep)
        final_df['channel_return_rate'] = final_df['Preferred_Shopping_Channels'].map(chan_ref)
        final_df['channel_exchange_rate'] = final_df['Preferred_Shopping_Channels'].map(chan_exch)
        final_df['channel_keep_rate'] = final_df['Preferred_Shopping_Channels'].map(chan_keep)
        self.model_solutions.dump_data(folders=path,file_name=file_name,obj=final_df)
        return final_df

    def encoding_data(self,data:pd.DataFrame=None, save:bool=False,path:Optional[List[str]]=None,file_name:Optional[str]=None):
        if data is None:
            data = self.data_preprocessing()

        class_counts = data['returned'].value_counts()
        min_class_size = class_counts.min()

        # Resample each class to match the smallest class
        balanced_dfs = []
        for cls in class_counts.index:
            resampled = resample(
                data[data['returned'] == cls],
                replace=False,
                n_samples=min_class_size,
                random_state=50
            )
            balanced_dfs.append(resampled)

        # Concatenate and shuffle
        balanced_data = pd.concat(balanced_dfs).sample(frac=1, random_state=50).reset_index(drop=True)

        # Split features and labels
        y = balanced_data['returned']
        X = balanced_data.drop(columns=['id','purchase_date','returned'])

        # One-hot encoding
        X = pd.get_dummies(X, drop_first=False)

        if save==True:
            if save==True and path is None:
                raise TypeError("Specify the Path Where the file is to be saved")
            else:
                save_data = X.copy()
                save_data['returned'] = y
                self.model_solutions.dump_data(folders=['data'] + path, file_name=file_name, obj=save_data)
                print("Data Saved Successfully ✔️")
        return X, y
    
    
    def schema(self, data_file:pd.DataFrame=None):
        print("Schema method called ✔️")
        if data_file is None:
            data_file, _ = self.encoding_data()

        schema = {}
        for idx,col in enumerate(data_file.columns):
            schema[col] = data_file[col].dtypes

        return schema
    
    def train_test_split(self,
                         X:Any=None,
                         Y:Any=None,
                         test_size:float=0.3,
                         save_encoders:bool=False,
                         save_data:bool=False,
                         model_folders:List[str]=['models','data_processing_models'],
                         encoder_names:List[str]=['standard_scaler.pkl','label_encoder.pkl'],
                         data_file_name:str='encoded_data.csv',
                         data_folders:List[str]=['data','processed','encoded']
                         ):
        

        if X is None and Y is None:

            X,Y = self.encoding_data()
            X_train,X_test,y_train,y_test = train_test_split(X,Y, stratify=Y, test_size=test_size,random_state=42)
            X_train['purchase_amount'] = self.scaler_encoder.fit_transform(X_train[['purchase_amount']])
            X_test['purchase_amount'] = self.scaler_encoder.transform(X_test[['purchase_amount']])
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

            if save_encoders==True and save_data==True:

                X_train_copy = X_train.copy()
                X_test_copy = X_test.copy()
                X_train_copy['returned'] = y_train
                X_test_copy['returned'] = y_test

                encoded_data = pd.concat([X_train_copy,X_test_copy],ignore_index=True)
                self.model_solutions.save_model(folders=model_folders,file_name=encoder_names[0],obj=self.scaler_encoder)
                self.model_solutions.save_model(folders=model_folders,file_name=encoder_names[1],obj=self.label_encoder)
                self.model_solutions.dump_data(folders=data_folders,file_name=data_file_name,obj=encoded_data)

            elif save_data:
                X_train_copy = X_train.copy()
                X_test_copy = X_test.copy()
                X_train_copy['returned'] = y_train
                X_test_copy['returned'] = y_test
                encoded_data = pd.concat([X_train_copy,X_test_copy],ignore_index=True)
                self.model_solutions.dump_data(folders=data_folders,file_name=data_file_name,obj=encoded_data)
            return X_train,X_test,y_train,y_test
        else:
            X_train,X_test,y_train,y_test = train_test_split(X,Y, stratify=Y, test_size=test_size,random_state=42)
            X_train['purchase_amount'] = self.scaler_encoder.fit_transform(X_train[['purchase_amount']])
            X_test['purchase_amount'] = self.scaler_encoder.transform(X_test[['purchase_amount']])
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            if save_encoders==True:
                self.model_solutions.save_model(folders=model_folders,file_name=encoder_names[0],obj=self.scaler_encoder)
                self.model_solutions.save_model(folders=model_folders,file_name=encoder_names[1],obj=self.label_encoder)
            return X_train,X_test,y_train,y_test