# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:03:38 2020

@author: User
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from data_print import pp_conf_matrix

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    
    
def GridSearchCV_Classifiers(X,y,models_params,scoring='accuracy', cv=5,n_jobs=-1,savefig_dir=None):
    
    models={}
    params={}
    for model in models_params.keys():
        models[model]=models_params[model][0]
        params[model]=models_params[model][1]
        
    GridSearch_helper = EstimatorSelectionHelper(models, params)
    GridSearch_helper.fit(X, y, scoring=scoring, n_jobs=n_jobs,cv=cv)
    results=GridSearch_helper.score_summary(sort_by='mean_score')
    results=results.reset_index(drop = True)
    
    print(results)

    ax=results[:10].plot.bar(x='estimator',y=['min_score', 'mean_score', 'max_score'],figsize=(20,10))
    ax.set_ylabel('Scores')
    ax.set_ylim(0.8, 1.05)
    ax.set_yticks(np.arange(0.5, 1.05, step=0.05))
    x_offset = 0.05
    y_offset = 0.005
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.3%}".format(b.y1 + b.y0)        
        ax.annotate(val, (b.x0 + x_offset, b.y1 + y_offset),rotation=90)
    if savefig_dir!=None:
        plt.savefig(savefig_dir, bbox_inches='tight') 
    plt.show()
      
    print('________________________________________________________________________')    
    
    
    
def Best_Classifiers(X_train, X_test, y_train, y_test,models_params,expl_lables,savefig_dir=None):
    
   
    params={}
    for model_name in models_params.keys():
        print (model_name)
        model=models_params[model_name][0]
        params=models_params[model_name][1]
        for param in params.keys():
            params[param]=params[param][0]
            
        model.set_params(**params)
        
        clf=model.fit(X_train, y_train)
        pp_conf_matrix(clf,{'Навчальна вибірка':(X_train,y_train),'Тестова вибірка':(X_test,y_test)},expl_lables=expl_lables,savefig_dir=savefig_dir)    
        print('________________________________________________________________________')  
        
        
        
        
def dict_to_models(models_params):
    models=[]
    for model_name in models_params.keys():
        model=models_params[model_name][0]
        params=models_params[model_name][1]
        for param in params.keys():
            params[param]=params[param][0]
            
        model.set_params(**params)
        models.append((model_name,model))
       
    return models       
        
        
        
        
        
        
        
           