class RFE:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import pickle
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    def rfefeature(indep_x,dep_y,n):
        rfelist=[]
        log_model=LogisticRegression(solver='lbfgs')
        svc=SVC(kernel = 'linear', random_state = 0)
        DT=DecisionTreeClassifier(criterion = 'gini', max_features='sqrt',splitter='best',random_state = 0)
        RF=RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        rfemodellist=[log_model,svc,DT,RF]
        for i in rfemodellist:
            print(i)
            log_rfe=RFE(estimator=i, n_features_to_select=n)
            log_fit=log_rfe.fit(indep_x,dep_y)
            log_rfe_feature=log_fit.transform(indep_x)
            rfelist.append(log_rfe_feature)
        return rfelist

    def split_scaler(indep_x,dep_y):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        x_train,x_test,y_train,y_test=train_test_split(indep_x,dep_y,test_size=0.25,random_state=42)
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test,y_train,y_test

    def cm_prediction(classifier,x_test):
        y_pred=classifier.predict(x_test)
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(y_test,y_pred)
        from sklearn.metrics import classification_report
        report=classification_report(y_test,y_pred)
        from sklearn.metrics import accuracy_score
        Accuracy=accuracy_score(y_test,y_pred)
        return classifier,Accuracy,report,cm,x_test,y_test

    def logistic(x_train,y_train,x_test):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,cm,x_test,y_test=cm_prediction(classifier,x_test)
        return classifier,Accuracy,report,cm,x_test,y_test

    def svc(x_train,y_train,x_test):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        param_grid={'kernel':['rbf','linear','poly'],'gamma':['auto','scale']}
        classifier=GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=3,n_jobs=-1,scoring='f1_weighted')
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,cm,x_test,y_test=cm_prediction(classifier,x_test)
        return classifier,Accuracy,report,cm,x_test,y_test

    def Decision(x_train,y_train,x_test):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        param_grid={'criterion':['gini','entropy','log_loss'],'splitter':['best','random']}
        classifier=GridSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=3,n_jobs=-1,scoring='f1_weighted')
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,cm,x_test,y_test=cm_prediction(classifier,x_test)
        return classifier,Accuracy,report,cm,x_test,y_test

    def random(x_train,y_train,x_test):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        param_grid={'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2'],'n_estimators':[100,200]}
        classifier=GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=3,n_jobs=-1,scoring='f1_weighted')
        classifier.fit(x_train,y_train)
        classifier,Accuracy,report,cm,x_test,y_test=cm_prediction(classifier,x_test)
        return classifier,Accuracy,report,cm,x_test,y_test

    def rfeclassification(acclog,accsvc,accdt,accrf):
        rfe_dataframe=pd.DataFrame(index=['Logistic','svc','DT','RF'],columns=['Logistic','SVC','DecisionTree','RandomForest'])
        for number,idex in enumerate(rfe_dataframe.index):
            rfe_dataframe['Logistic'][idex]=acclog[number]
            rfe_dataframe['SVC'][idex]=accsvc[number]
            rfe_dataframe['DecisionTree'][idex]=accdt[number]
            rfe_dataframe['RandomForest'][idex]=accrf[number]
        return rfe_dataframe

    dataset=pd.read_csv("preprocessed ILPD.csv")
    df=dataset
    df=pd.get_dummies(df,dtype=int,drop_first=True)
    df
    indep_x=df.drop('Selector',axis=1)
    dep_y=df['Selector']
    indep_x.shape
    dep_y.shape
    print(indep_x.shape)
    print(type(indep_x))

    rfelist=rfefeature(indep_x,dep_y,4)
    acclog=[]
    accsvc=[]
    accdt=[]
    accrf=[]
    for i in rfelist: 
        print(type(i))
        x_train, x_test, y_train, y_test=split_scaler(i,dep_y)   
        classifier,Accuracy,report,cm,x_test,y_test=logistic(x_train,y_train,x_test)
        acclog.append(Accuracy)
        classifier,Accuracy,report,cm,x_test,y_test=svc(x_train,y_train,x_test)
        accsvc.append(Accuracy)
        classifier,Accuracy,report,cm,x_test,y_test=Decision(x_train,y_train,x_test)
        accdt.append(Accuracy)
        classifier,Accuracy,report,cm,x_test,y_test=random(x_train,y_train,x_test)
        accrf.append(Accuracy)
    result=rfeclassification(acclog,accsvc,accdt,accrf)

    #shows the selected features
    def rfefeature(indep_x,dep_y,n):
        rfelist=[]
        log_model=LogisticRegression(solver='lbfgs')
        svc=SVC(kernel = 'linear', random_state = 0)
        DT=DecisionTreeClassifier(criterion = 'gini', max_features='sqrt',splitter='best',random_state = 0)
        RF=RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        rfemodellist=[log_model,svc,DT,RF]
        for i in rfemodellist:
            print(i)
            log_rfe=RFE(estimator=i, n_features_to_select=n)
            log_fit=log_rfe.fit(indep_x,dep_y)
            log_rfe_feature=log_fit.transform(indep_x)
            mask=log_fit.get_support()
            selected_features=indep_x.columns[mask].tolist()
            print(selected_features)
            rfelist.append(log_rfe_feature)
        return selected_features
