import requests
import pickle

input_file = './model/ridge_model.bin'

with open(input_file, 'rb') as f_in: 
    model, test_data = pickle.load(f_in)

test_data = test_data.tolist()

url = 'http://food-delivery-model-env.eba-mkkwnvyp.us-east-2.elasticbeanstalk.com/predict'

test_data_json = [[float(x) if isinstance(x, (int, float)) else bool(x) for x in row] for row in test_data]

print(requests.post(url, json=test_data_json).json())


