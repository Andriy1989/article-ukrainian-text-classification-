# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA,KernelPCA,TruncatedSVD
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def data_class_display(dataframe,class_column,expl_lables):
    """ data_class_display - shows the class shares in the dateframe. 
    
    Keyword arguments:
    dataframe -- dateframe with classes to display
    class_column -- column with class labels
    expl_lables -- dictionary for explaining class labels    
    
    """
    
    data=dataframe[class_column].value_counts()
    sum_value=data.sum()
    data=data.T   
    
    labels=[]
    fracs=[]
    explode=[]
    
    for index,value in data.iteritems():
        labels.append('{0}.{1} - {2:.2f} % ({3} екземплярів)'.format(index,expl_lables[index],(value/sum_value*100),value))
        fracs.append(value)
        explode.append(0.1)
        
    fig, axs = plt.subplots()
    axs.pie(x=fracs, autopct='%1.2f%%',radius=1.0,wedgeprops=dict(width=0.7, edgecolor='w'))
    plt.legend(fracs, labels=labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    axs.set_title("Класи")
    plt.show()
    
    
def plot_2d_decision_regions(X,y,classifier,resolution=0.02,expl_lables=None,savefig_dir=None): 
    """ plot_2d_decision_regions - displays classes and decision regions in 2-dimensional space. 
    
    Keyword arguments:
    X -- training vector
    y -- target vector
    resolution -- spacing between values. For any output out, this is the distance between two adjacent values(default 0.02)
    expl_lables -- dictionary for explaining class labels (default None)
    
    """
    
    markers=('s' , 'x', 'o', '^', 'v', '*', 'h')
    colors=('red','blue','lightgreen','gray','cyan','yellow','black')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4 , cmap=cmap)
  
    
    for idx,cl in enumerate(np.unique(y)):
        if expl_lables:
            lable_cl=expl_lables[cl]
        else:
            lable_cl=cl
        plt.scatter(x=X[y==cl,0], y=X[y==cl, 1],alpha=0.8, c=cmap(idx), marker=markers[idx], label=lable_cl)
    plt.xlabel('PC1')
    plt.ylabel('РС2')
        
    plt.legend(loc= 'upper right')
    if savefig_dir!=None:
        plt.savefig(savefig_dir)
    plt.show()  
    
def plot_2d_labled_data(X,y,expl_lables=None,savefig_dir=None): 
    """ plot_2d_labled_data - displays classes in 2-dimensional space. 
    
    Keyword arguments:
    X -- training vector
    y -- target vector   
    expl_lables -- dictionary for explaining class labels (default None)
    
    """
    
    markers=('s' , 'x', 'o', '^', 'v', '*', 'h')
    colors=('red','blue','lightgreen','gray','cyan','yellow','black')
    cmap = ListedColormap(colors[:len(np.unique(y))])
        
    for idx,cl in enumerate(np.unique(y)):
        if expl_lables:
            lable_cl=expl_lables[cl]
        else:
            lable_cl=cl
        plt.scatter(x=X[y==cl,0], y=X[y==cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=lable_cl)
    plt.xlabel('PC1')
    plt.ylabel('РС2')
        
    plt.legend(loc= 'upper right')
    if savefig_dir!=None:
        plt.savefig(savefig_dir)    
    plt.show()    

    
def plot_3d_labled_data(X,y,elev_step=360,azim_step=60,expl_lables=None,savefig_dir=None): 
    """ plot_3d_labled_data - displays classes in 3-dimensional space. 
    
    Keyword arguments:
    X -- training vector
    y -- target vector    
    expl_lables -- dictionary for explaining class labels (default None)
    
    """
    
    axs=[]
    elev_step_number=math.ceil(360/elev_step)
    azim_step_number=math.ceil(360/azim_step)
    index_number=elev_step_number*azim_step_number
    nrows=math.ceil(index_number/3)
    ncols=math.ceil(index_number/nrows)  
    
    if nrows<=3:
        fig = plt.figure(figsize=(10*ncols/nrows,10))
    else:
        fig = plt.figure(figsize=(10,10*nrows/ncols))
        
    index=0    
    elev=30
    while elev<390:
        azim=-60
        while azim<300:
            index+=1
            ax=fig.add_subplot(nrows,ncols, index, projection='3d')
            ax.view_init(elev, azim)
            axs.append(ax)
            azim+=azim_step
        elev+=elev_step
    
    markers=('s' , 'x', 'o', '^', 'v', '*', 'h')
    colors=('red','blue','lightgreen','gray','cyan','yellow','black')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    fontP = FontProperties()
    
    if index_number<=3:
        fontP.set_size('large')
    else:    
        fontP.set_size('small')
        
    for ax in (axs):
        for idx,cl in enumerate(np.unique(y)):
            if expl_lables:
                lable_cl=expl_lables[cl]
            else:
                lable_cl=cl
            ax.scatter(xs=X[y==cl,0], ys=-X[y==cl, 1], zs=X[y==cl, 2],
            alpha=0.8, c=cmap(idx),
            marker=markers[idx], label=lable_cl) 
        ax.legend(loc='best', bbox_to_anchor=(1.2, 1.0), prop=fontP)
    if savefig_dir!=None:
        plt.savefig(savefig_dir)        
    plt.show()    
        

def PCA_text_data_display(X,y,stop_words=None,tokenizer=None,Vectorizer=CountVectorizer,expl_lables=[],n_components=2,feature_selection=False,
                          selector=SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=500),threshold=0.000001),
                          decision_regions=False,elev_step=360,azim_step=120,savefig_dir=None):
    """ PCA_text_data_display - displays text_data in 2- or 3-dimensional space. 
    
    Keyword arguments:
    X -- training text data vector
    y -- target vector    
    stop_words -- list of stop words (default None)
    tokenizer -- Override the string tokenization step while preserving the preprocessing and n-grams generation steps (default None)
    Vectorizer -- dictionary for explaining class labels (default CountVectorizer)
    expl_lables -- dictionary for explaining class labels (default None)
    n_components -- number of components to keep. n_components must be 2 or 3 (default 2) 
    feature_selection -- if True uses feature selection before PCA (default False)
    selector -- selector for feature selection. Ignored when feature_selection=False (default SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=500),threshold=0.000001)).
    decision_regions -- displays classes and decision regions in 2-dimensional space. Ignored when n_components!=2 (default False). 
    elev_step -- This can be used to rotate the axes programatically. elev_step stores step for the elevation angle in the z plane. Ignored when n_components!=3 (default 360). 
    azim_step -- This can be used to rotate the axes programatically. elev_step stores step for the azimuth angle in the x,y plane. Ignored when n_components!=3 (default 120).
    savefig_dir -- A path to save the current figure (default None)
    """
        
    print('PCA')
    vectorizer=Vectorizer(tokenizer=tokenizer,stop_words=stop_words)
    pca=PCA(n_components=n_components)
    
    if feature_selection:
        
        pipe=Pipeline([('Vectorizer',vectorizer),
                       ('feature_selection',selector)])
    else:
        pipe=Pipeline([('Vectorizer',vectorizer)])
        
    X=pipe.fit_transform(X, y).toarray() 
    X=pca.fit_transform(X)
    if n_components==2:
        if decision_regions:
            lr = LogisticRegression()
            lr.fit(X, y)
            plot_2d_decision_regions(X,y,classifier=lr,expl_lables=expl_lables,savefig_dir=savefig_dir)
        else:
            plot_2d_labled_data(X,y,expl_lables=expl_lables,savefig_dir=savefig_dir)
    elif n_components==3: 
        plot_3d_labled_data(X,y,expl_lables=expl_lables,elev_step=elev_step,azim_step=azim_step,savefig_dir=savefig_dir)
    else:
        print('n_components must be 2 or 3') 

    print('________________________')   
    
  
    
    
def pp_conf_matrix(classifier,X_y_dict,expl_lables,savefig_dir=None):
    """ pp_conf_matrix - display confusion matrix, accuracy, recall, precision and f1 score to evaluate the accuracy of a classification. 
    
    Keyword arguments:
    classifier -- dateframe with classes to display
    X_y_dict -- dictionary type: {dataset name: (X column, y column)}
    expl_lables -- dictionary for explaining class labels    
    savefig_dir -- A path to save the current figure (default None)
    
    """
    
    
    figsize=max(10,2*len(expl_lables))
    list_keys=list(X_y_dict.keys())
    fig,axs=plt.subplots(nrows=1, ncols=len(list_keys),figsize=(figsize,figsize), constrained_layout=True)
    
    if len(expl_lables)==2:
        average='binary'
    else:
        average='macro'
    
    for key in list_keys:
        X_test=X_y_dict[key][0]
        y_pred=classifier.predict(X_test)
        y_test=X_y_dict[key][1]
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        
        ax=axs[list_keys.index(key)]
        
        im=ax.matshow(confmat, cmap=plt.cm.prism, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
        
       
        text='{0}\nТочність: {1}\nRecall score: {2}\nPrecision_score: {3}\nF1 score(macro): {4}'.format(key,
                                                                                  classifier.score(X_test,y_test),
                                                                                  recall_score(y_true=y_test, y_pred=y_pred,average=average),
                                                                                  precision_score(y_true=y_test, y_pred=y_pred,average=average),
                                                                                  f1_score(y_true=y_test, y_pred=y_pred,average=average))
        ax.text(0.5, -0.33*10/figsize, text,
         horizontalalignment='center',
         fontsize=11,
         transform = ax.transAxes)
                
        ax.set_xticklabels(['']+list(expl_lables.values()),rotation=90)
        ax.set_yticklabels(['']+list(expl_lables.values()))  
        
        axins = inset_axes(ax,
                   width="10%",  
                   height="100%", 
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
        fig.colorbar(im,cax=axins,ax=axs[list_keys.index(key)])
        fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
        hspace=0.2, wspace=0.2)
        ax.set_xlabel('Розпізнані мітки' )
        ax.set_ylabel('Вірні мітки')
    if savefig_dir!=None:
        plt.savefig(savefig_dir)
    plt.show()   
     