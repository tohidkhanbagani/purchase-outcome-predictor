from src.data_ingestion.data_ingest import Ingest
from src.utils.functions import ModelSolutions
from typing import List,Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict



class InputModelFeatures:

    def __init__(self):
        
        self.ingest = Ingest()
        self.model_solutions = ModelSolutions()
        
        self.standard_encoder = self.model_solutions.load_model(folders=['models','data_processing_models'],file_name='standard_scaler.pkl')
        self.label_encoder = self.model_solutions.load_model(folders=['models','data_processing_models'],file_name='label_encoder.pkl')
        
        self.predictive_encoded_data = self.ingest.data_ingest(folders=['data','processed','encoded'],file_name='encoded_data.csv')
        self.predictive_encoded_data_schema = self.ingest.schema(obj=self.predictive_encoded_data,choice='dictionary')
        
        self.processed_data_metrics = self.ingest.data_ingest(folders=['data','processed','processed_data'],file_name='processed_metrics.csv')

        self.product_list = list(self.processed_data_metrics['product'].unique())
        self.product_to_index = {product: i for i, product in enumerate(self.product_list)}

        self.channels = list(self.processed_data_metrics['Preferred_Shopping_Channels'].unique())


    def predictive_model_features(
        self,
        id: str,
        product_name: str,
        preferred_shopping_channel: str,
        status: List[str]
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # --- 1. Binning setups ---
        segment_labels = ['one_time_buyer', 'infrequent_buyer', 'frequent_buyer', 'loyal_buyer']
        segment_bins   = [0, 2, 4, 10, float('inf')]
        amt_labels     = ['very_low', 'low', 'mid', 'high', 'very_high', 'VIP']
        amt_bins       = [0, 10, 50, 200, 500, 1000, float('inf')]

        # --- 2. Default ID and timestamp ---
        if id is None:
            id = 'CUS' + str(np.random.randint(10_000, 1_000_000))
        current_date = pd.to_datetime(datetime.now())

        if not isinstance(status, list):
            status = [status]
        # --- 3. Compute user-level stats ---
        user_purchases    = len(status)
        refunds           = status.count('refund')
        exchanges         = status.count('exchange')
        keeps             = status.count('no')

        user_return_rate   = (refunds   / user_purchases) * 100
        user_exchange_rate = (exchanges / user_purchases) * 100
        user_keep_rate     = (keeps     / user_purchases) * 100

        # --- 4. Look up category-level rates ---
        cat_row = self.processed_data_metrics.query("product == @product_name").iloc[0]
        cat_name      = cat_row['category']
        cat_ref_rate  = cat_row['cat_refund_rate']
        cat_exc_rate  = cat_row['cat_exchange_rate']
        cat_keep_rate = cat_row['cat_kept_rate']
        purchase_amount = cat_row['purchase_amount']

        # --- 5. Segment and amount bins ---
        segment = pd.cut([user_purchases], bins=segment_bins, labels=segment_labels, right=False)[0]
        amt_bin = pd.cut([purchase_amount], bins=amt_bins, labels=amt_labels)[0]


        # --- 6. Look up channel-level rates, with fall-back ---
        chan_df = self.processed_data_metrics.query(
            'product == @product_name and Preferred_Shopping_Channels == @preferred_shopping_channel'
        )
        if not chan_df.empty:
            chan_ref_rate, chan_exch_rate, chan_keep_rate = chan_df.iloc[0][
                ['channel_return_rate','channel_exchange_rate','channel_keep_rate']
            ]
        else:
            # Fallback to category averages if channel not found
            chan_ref_rate, chan_exch_rate, chan_keep_rate = (
                cat_ref_rate,
                cat_exc_rate,
                cat_keep_rate
            )

        # --- 7. Assemble one-row DataFrame of “raw” features ---
        row = {
            'id':                          id,
            'category':                    cat_name,
            'product':                     product_name,
            'purchase_date':               current_date,
            'purchase_amount':             purchase_amount,
            'returned':                    status,
            'Preferred_Shopping_Channels': preferred_shopping_channel,
            'purchases':              user_purchases,
            'return_rate':            user_return_rate,
            'exchange_rate':          user_exchange_rate,
            'kept_rate':              user_keep_rate,
            'segment':                     segment,
            'days_since_last':    0,        #  it will be overrid later
            'amt_bin':                     amt_bin,
            'cat_refund_rate':             cat_ref_rate,
            'cat_exchange_rate':           cat_exc_rate,
            'cat_kept_rate':               cat_keep_rate,
            'channel_return_rate':            chan_ref_rate,
            'channel_exchange_rate':          chan_exch_rate,
            'channel_keep_rate':              chan_keep_rate
        }
        completed_data = pd.DataFrame([row])

        # --- 8. Encode & align to your predictive schema ---
        encoded = completed_data.drop(columns=['id','purchase_date','returned'])
        encoded[['purchase_amount']] = self.standard_encoder.transform(
            encoded[['purchase_amount']]
        )

        dummies    = pd.get_dummies(encoded)
        final_input = pd.concat([
            pd.DataFrame(columns=self.predictive_encoded_data_schema.keys()),
            dummies
        ], ignore_index=True).fillna(0)

        # enforce dtypes and drop any leftover
        for col, dtype in self.predictive_encoded_data_schema.items():
            final_input[col] = final_input[col].astype(dtype)
        final_input.drop(columns=['returned'], errors='ignore', inplace=True)

        completed_data['returned'] = completed_data['returned'][0][0]

        return final_input, completed_data





    def bulk_entries(
        self,
        entries:int=50,
        min_id:int=10000,
        max_id:int=20000,
        replace:bool=True,
        date:List[int]=[2024,1,1],
        date_entries:int=50,
        max_days:int=100
        )->pd.DataFrame:
            num_entries = entries

            ids = np.random.choice([np.random.randint(1000,2000) for i in range(num_entries)],size=num_entries,replace=replace)
            products = np.random.choice(self.product_list,size=num_entries,replace=True)
            preferred_shopping_channel = np.random.choice(list(self.processed_data_metrics['Preferred_Shopping_Channels'].unique()),size=num_entries,replace=True)
            status = np.random.choice(list(self.processed_data_metrics['returned'].unique()),size=num_entries,replace=True)
            dates = np.random.choice([pd.to_datetime(
            datetime(
                *date,np.random.randint(0,23),
                np.random.randint(0,59),
                np.random.randint(0,59)) + timedelta(days=np.random.randint(0,max_days))) for i in range(date_entries)
                ],size=num_entries,replace=True)
            
            

            test_data = pd.DataFrame({
                'id':ids,
                'products':products,
                'Preffered_Shopping_Channels':preferred_shopping_channel,
                'status':status,
                'purchase_date':dates
                }).sort_values(by='purchase_date',ascending=True)
            test_data['days_since_last'] = test_data.groupby('id')['purchase_date'].diff().dt.days.fillna(0)
            return test_data




    def bulk_encoder(self, data:pd.DataFrame= None)-> Tuple[pd.DataFrame, pd.DataFrame]:
        if data is None:
            data = self.bulk_entries()
        status_history = defaultdict(list)
        final_inputs = []
        data_final = []

        # Tracker for each customer's past statuses
        status_history = defaultdict(list)

        final_inputs = []
        data_final = []

        for i, row in data.iterrows():
            current_id = row['id']
            product = row['products']
            chan = row['Preffered_Shopping_Channels']
            stat = row['status']
            days_last = row['days_since_last']

            # Include history + current
            status_so_far = status_history[current_id] + [stat]

            final_input, raw_data = self.predictive_model_features(
                id=current_id,
                product_name=product,
                preferred_shopping_channel=chan,
                status=status_so_far
            )

            final_input['days_since_last'] = days_last
            raw_data['days_since_last'] = days_last

            final_inputs.append(final_input)
            data_final.append(raw_data)

            # Update history
            status_history[current_id].append(stat)

        final_data = pd.concat(final_inputs, ignore_index=True)
        final_data_1 = pd.concat(data_final, ignore_index=True)
        final_data_1['returned'] = data['status']

        return final_data, final_data_1