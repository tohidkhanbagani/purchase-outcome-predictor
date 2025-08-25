from xgboost import XGBClassifier
from typing import List
from typing import Optional
import warnings
from sklearn.metrics import accuracy_score
from src.utils.functions import ModelSolutions
from src.data_ingestion.data_ingest import Ingest
from src.features.feature_builder import PredictiveModelData

warnings.filterwarnings('ignore')



class TrainPipeline:

    def __init__(self,
                 data_folder:list[str]=['data','raw'],
                 data_file:str='E-Commerce_data.csv',
                 max_depth:int=5,
                 n_estimators:int=200,
                 colsample_bytree:float=0.8,
                 subsample:float=0.8
                 ):
        self.ingest = Ingest()
        self.model_solutions = ModelSolutions()
        self.raw_data = self.ingest.data_ingest(folders=data_folder,file_name=data_file)
        self.predictor_data_init = PredictiveModelData(data_frame=self.raw_data)
        self.predictive_model = XGBClassifier(max_depth=max_depth,n_estimators=n_estimators,colsample_bytree=colsample_bytree,subsample=subsample)


    def training_predictive_model(self,save_path:Optional[List[str]]=['models','trained_models'],production:bool=False,model_name:str='xgboost_model.pkl'):
        X_train,X_test,y_train,y_test = self.predictor_data_init.train_test_split(test_size=0.3,save_data=True,save_encoders=True)
        self.predictive_model.fit(X_train,y_train)
        model_predictions = self.predictive_model.predict(X_test)
        accuracy = round(accuracy_score(y_pred=model_predictions,y_true=y_test),2)*100
        if production:
            if accuracy>=65:
                self.model_solutions.save_model(folders=save_path,file_name=model_name,obj=self.predictive_model)
                return print('the model accuracy is :- ',accuracy,'\n','and the model is saved at models/tarined_models/xgboost_model.pkl')
            else:
                return print(f"the models accuracy :- {accuracy} whcih is quite low and it needs more working")
        else:
            model_predictions, y_test