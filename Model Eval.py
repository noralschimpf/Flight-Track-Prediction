import pandas as pd
import numpy as np
from model import CONV_LSTM, load_model
from custom_dataset import CustomDataset, ToTensor
# Load model and specified set of test flights
# For each Model
    # Collect Predicted and Actual Trajectory into flight-specific CSV
    # Store loss (MSE, rmse,mae, mape) in general evaluation CSV

root = 'data/'
output = 'Output/'
dev = 'cpu'

# open model and data
mdl = load_model('Models/CONV-LSTM-OPTAdam-LOSSMSELoss()-EPOCHS5-LSTM6_100_2')
test_flights = pd.read_csv('test_flight_samples.txt').values.squeeze()
flight_data = CustomDataset(root_dir=root, transform=ToTensor(), device=dev)
flight_data.validate_sets(underMin=100)

# begin validation
mdl.evaluate(flight_data, test_flights)