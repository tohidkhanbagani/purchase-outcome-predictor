import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Annotated
import json


class PurchasePredictionInput(BaseModel):
    id: Annotated[str, Field(..., description="Customer ID")]
    product_name: Annotated[str, Field(..., description="Product name")]
    preferred_shopping_channel: Annotated[str, Field(..., description="Preferred shopping channel")]
    status: Annotated[List[str], Field(..., description="List of status history e.g. ['refund', 'no']")]


class BulkPredictionInput(BaseModel):
    n_entries:Annotated[int, Field(..., description='Number of Entries you want in the data')] = 50
    replace:Annotated[bool, Field(..., description='Do you want to stimulate repeated buying behaviour?')] = True
    date:Annotated[List[int], Field(..., description='the date from which you want to stimulate teh transactions')] = [2024,1,1]
    date_entries:Annotated[int, Field(..., description='the ')]=50
    max_days:Annotated[int, Field(..., description='number of days from the start date')]=10