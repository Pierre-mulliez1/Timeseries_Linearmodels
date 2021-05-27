---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Time series in Python using linear models
**Author**: Pierre Mulliez
**contact**: pierremulliez1@gmail.com **Last edited**:27/05/2021

```python
#Importing packages 
import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression,BayesianRidge,Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
```

```python
#time series train test split 
result_df = pd.DataFrame(columns = {'model','score_mean', 'score_deviation'})
X = day_df.loc[:,day_df.columns != 'cnt']
y = day_df.loc[:, 'cnt']
print(X.head())
print('')
print(y.head())
```

```python
count = 0
mod = ["Ridge ","Linear regression ", "Lasso ","Bayesian Ridge "]
for model in [Ridge(normalize=True),LinearRegression(),Lasso(),BayesianRidge()]:
    tscv = TimeSeriesSplit(n_splits = 2)
    #Corss validation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        result_df = result_df.append({
                                         "model": mod[count],
                                          "score_mean": abs(scores.mean()),
                                          'score_deviation': scores.std()
                                         },ignore_index=True) 
    count += 1
```

```python
aggresults = result_df.groupby('model').agg({'score_mean':'mean','score_deviation': 'mean'})
#visualise the best models for further use
bestmod = aggresults.loc[aggresults['score_mean'] == aggresults['score_mean'].min(),:]
print("The best model score without optimization is {}".format(bestmod.iloc[0,0]))
bestmod
```

**Optimization linear models**

```python
bresult_df = pd.DataFrame(columns = {'model','score_mean', 'score_deviation','predctionrMSE'})
count = 0
mod = ["Ridge ","Linear regression ", "Lasso ","Bayesian Ridge "]
#Hyperparameters for optimisation
hyperparameters = {'alpha':[1, 0.5 , 1.5, 2,0.1]}
bayeshyperparameters = {'alpha_1':[1, 0.5 , 1.5, 2,0.1], 'lambda_1':[1, 0.5 , 1.5, 2,0.1]}
for model in [Ridge(normalize=True),LinearRegression(),Lasso(),BayesianRidge()]:
    tscv = TimeSeriesSplit(n_splits = 2)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if mod[count] == 'Linear regression ':
            #no hyperparameters
            modelbest = model
        elif mod[count] == 'Bayesian Ridge ':
            modelbest = GridSearchCV(model,bayeshyperparameters,cv = tscv)
        else:
            modelbest = GridSearchCV(model,hyperparameters,cv = tscv)
        scores = cross_val_score(modelbest, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        #Prediction
        modelbest.fit(X_train, y_train)
        y_pred = modelbest.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        bresult_df = bresult_df.append({
                                         "model": mod[count],
                                          "score_mean": abs(scores.mean()),
                                          'score_deviation': scores.std(),
                                            'predctionrMSE': error
                                         },ignore_index=True) 
    count += 1
```

```python
aggresultso = bresult_df.groupby('model').agg({'score_mean':'mean','score_deviation': 'mean','predctionrMSE':'mean'})
aggresultso
```

```python
bestmodo = aggresultso.loc[aggresultso['score_mean'] == aggresultso['score_mean'].min(),:]
print("The best model score with optimization is {}".format(bestmodo.iloc[0,0]))
bestmodo
```
