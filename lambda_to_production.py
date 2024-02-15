import pickle
import pandas as pd

def predict_new_data(new_lat,new_lon):
    try:
        filename = 'finalized_model_KNN.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        d = {'lat':new_lat , 'lon': new_lon}
        df_new = pd.DataFrame(data=d,index=[0])
        prediction=loaded_model.predict(df_new)
        return prediction[0]
    except:
        return 6