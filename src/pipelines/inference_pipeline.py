from src.models.sales_predicter.purchase_outcome import PurchaseOutcomeModel
from src.data_ingestion.data_ingest import Ingest
from src.features.inference_feature_builder import InputModelFeatures
from typing import List
import pandas as pd
import numpy as np

class ModelInference:

    def __init__(self):
        self.main_data = Ingest()
        self.model_inputs = InputModelFeatures()
        self.predictive_model = PurchaseOutcomeModel()
        self.product_list_array = np.array(self.model_inputs.product_list)

    
            
    def predictive_model_predictions(self, 
                                     multiple:bool=False,
                                     id: str=None, 
                                     product_name: str=None, 
                                     preferred_shopping_channel: str=None, 
                                     status: List[str]=None, 
                                     data:pd.DataFrame=None,
                                     n_entries:int=None,
                                     replace:bool=True,
                                     date:List[int]=[2024,1,1],
                                     date_entries:int=50,
                                     max_days:int=100
                                     ):
        if multiple:
            if n_entries is None:
                 encoded_data, feature_data = self.model_inputs.bulk_encoder(data=data) 
            else:
                 test_data = self.model_inputs.bulk_entries(entries=n_entries, replace=replace, date=date, date_entries=date_entries,max_days=max_days)
                 encoded_data, feature_data = self.model_inputs.bulk_encoder(data=test_data)


            model_predictions = self.model_inputs.label_encoder.inverse_transform(self.predictive_model.predict(encoded_data))
            feature_data['predictions'] = model_predictions
            feature_data['purchase_date'] = feature_data['purchase_date'].astype(str) # Convert to string
            feature_data = feature_data[['id','product','category','purchase_amount','purchase_date','Preferred_Shopping_Channels','returned','predictions']].copy()
            return feature_data
        else:
            encoded_data, feature_data = self.model_inputs.predictive_model_features(id=id,
                                                                                     product_name=product_name,
                                                                                     preferred_shopping_channel=preferred_shopping_channel,
                                                                                     status=status
                                                                                     )
            model_predictions = self.model_inputs.label_encoder.inverse_transform(self.predictive_model.predict(encoded_data))
            feature_data['predictions'] = model_predictions
            feature_data['purchase_date'] = feature_data['purchase_date'].astype(str) # Convert to string
            feature_data = feature_data[['id','product','category','purchase_amount','purchase_date','Preferred_Shopping_Channels','returned','predictions']].copy()
            return feature_data