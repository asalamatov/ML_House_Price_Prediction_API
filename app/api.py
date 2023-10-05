# pip install unicorn fastapi pydantic

from fastapi import FastAPI
from pydantic import BaseModel
import dill, pickle
from utils import Preprocessor
import pandas as pd


# # create api
app = FastAPI()

# # load GB model
with open('gb.pkl', 'rb') as f:
    model = dill.load(f)

# # type checking class thru pydantic
class ScoringItem(BaseModel):
    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs: float
    PostCode: str

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {
        'prediction': int(yhat)
    }


# # cd app
# # uvicorn api:app --reload

# import dill, pandas as pd, pickle
# with open('gb.pkl', 'rb') as f:
#     reloaded_model = pickle.load(f)
# item = {"TransactionDate":"2020.12","HouseAge":17.0,"DistanceToStation":467.6447748,"NumberOfPubs":4.0,"PostCode":"5222.0"}
# print(reloaded_model.predict(pd.DataFrame([item.values()], columns=item.keys())))