from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.pipelines.inference_pipeline import ModelInference
from api.input_classes import PurchasePredictionInput, BulkPredictionInput
from starlette.middleware.cors import CORSMiddleware

app =  FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def load_infrence():
    global inference
    print("Initializing Infrence...")
    infrence = ModelInference()
    print("Initialization Finished...")
    return infrence

infrence = load_infrence()

@app.post('/predict/purchase-outcome')
async def post_predict_outcome(user_inputs:PurchasePredictionInput):
    global inference
    result = infrence.predictive_model_predictions(
        id=user_inputs.id,
        product_name=user_inputs.product_name,
        preferred_shopping_channel=user_inputs.preferred_shopping_channel,
        status=user_inputs.status)
    prediction = result['predictions'].tolist()[0]
    
    return JSONResponse(status_code=200, content=prediction)



@app.post('/predict/bulk-purchase-predict')
async def post_predict_outcome_bulk(params:BulkPredictionInput):
    global inference
    result_data = infrence.predictive_model_predictions(
        multiple=True, 
        n_entries=params.n_entries, 
        replace=params.replace, 
        date=params.date, 
        date_entries=params.date_entries, 
        max_days=params.max_days
        )
    # Convert Timestamp objects to string before returning
    if 'purchase_date' in result_data.columns:
        result_data['purchase_date'] = result_data['purchase_date'].astype(str)
    return JSONResponse(status_code=200,content=result_data.to_dict('records'))


@app.get("/products")
def get_all_products():
    """
    Returns a list of all products with their details.
    """
    global infrence

    all_products_data = infrence.model_inputs.product_list

    return JSONResponse(content=all_products_data)

@app.get("/channels")
def get_all_channels():
    """
    Returns a list of all channels.
    """
    global infrence

    all_channels_data = infrence.model_inputs.channels

    return JSONResponse(content=all_channels_data)



