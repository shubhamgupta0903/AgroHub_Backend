
from pydantic import BaseModel
class BankNote(BaseModel):
    temperature: float 
    humidity: float 
    rainfall: float 

class Wheat(BaseModel):
    arrival_date: int
    tavg: float
    prcp: float