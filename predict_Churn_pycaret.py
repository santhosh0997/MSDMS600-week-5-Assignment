import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(file_path):
    df = pd.read_csv(file_path, index_col = 'customerID')
    return df

def make_predictions(df):
    model = load_model('pycaret_model')
    predictions = predict_model(model, data= df)
    predictions.rename({'prediction_label':'predicted_churn'}, axis=1, inplace = True)
    predictions['predicted_churn'].replace({0:'No Churn', 1:'Churn'}, inplace = True)
    return predictions['predicted_churn']
def probabilities(df):
    model = load_model('pycaret_model')
    predictions = predict_model(model,data=df)
    return predictions['prediction_score']

def predict_true(df):
    model = load_model('pycaret_model')
    predictions = predict_model(model, data= df)
    return predictions['prediction_label']


if __name__=='__main__':
    file_name = input("Give a file name:")
    df = load_data(file_name)
    print()
    print("predictions")
    print(f'output: {make_predictions(df)}')
    print()
    print("Probabilities")
    print(f'probability: {probabilities(df)}')
    print()
    print("True_values")
    print(predict_true(df))
    