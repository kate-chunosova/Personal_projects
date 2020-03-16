from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FlagColumns(BaseEstimator, TransformerMixin):
    """Feature creation based on columns that had high number of nulls.
    
    Flag columns match those columns that had high number of nulls (98%-100%) and contained potentially valuable information about class 1. Flag column is     equal to 1 if the data in the feature was absent and 0 otherwise.
    
    Args:
        flag(bool): if True creates feature representing flag columns, default True.
        
    Returns: 
        pd.DataFrame: transformed pandas DataFrame.
    """
    
    def __init__(self, flag=True):
        self.flag = flag
        
    def fit(self,X,y=None):    
        return self
    
    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame)
        
        try: 
            if self.flag == True:
                missing_numerical_columns = [i for i in X.iloc[:, :189].columns if (X[i].isna().sum()/len(X[i]) >= 0.3)]
            
                for i in missing_numerical_columns:
                    X.loc[:,i + '_flag'] = np.where(X.loc[:,i].isna() == True, 1, 0)
                
            return X

        except KeyError:
            cols_error = list(set(missing_numerical_columns) - set(X.columns))
            raise KeyError('[FlagColumns] DataFrame does not include the columns:', cols_error)
            