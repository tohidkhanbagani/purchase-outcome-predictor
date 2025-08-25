from src.utils.functions import ModelSolutions
from typing import List
from xgboost import XGBClassifier


class PurchaseOutcomeModel:

    def __init__(self,model_path:List[str]=['models','trained_models'],model_name:str='xgboost_model.pkl',gid_search:bool=False):
        self.model_solutions = ModelSolutions()
        if gid_search:
            self.model = self.model_solutions.load_model(folders=model_path,file_name=model_name)
            self.model = XGBClassifier(**self.model.best_params_)
        else:
            self.model = self.model_solutions.load_model(folders=model_path,file_name=model_name)
            
    def predict(self,data):
        model_prediction = self.model.predict(data)
        return model_prediction
