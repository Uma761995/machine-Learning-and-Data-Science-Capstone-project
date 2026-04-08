class Univariate:
    def quanqual(dataset):
        quan=[]
        qual=[]
        for columnName in dataset.columns:
            if (dataset[columnName].dtype=='O'):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan,qual

    def univariate(dataset,quan):
        descriptive=pd.DataFrame(index=['mean','median','mode','Q1:25%','Q2:50%','Q3:75%','Q:99%','Q4:100%','IQR','1.5rule','lesser','greater','min','max','skew','kurtosis','std','var'],columns=quan)
        for columnName in quan:
            descriptive.loc['mean',columnName]=dataset[columnName].mean()
            descriptive.loc['median',columnName]=dataset[columnName].median()
            descriptive.loc['mode',columnName]=dataset[columnName].mode()[0]
            descriptive.loc['Q1:25%',columnName]=dataset.describe()[columnName]['25%']
            descriptive.loc['Q2:50%',columnName]=dataset.describe()[columnName]['50%']
            descriptive.loc['Q3:75%',columnName]=dataset.describe()[columnName]['75%']
            descriptive.loc['Q:99%',columnName]=np.percentile(dataset[columnName],99)
            descriptive.loc['Q4:100%',columnName]=dataset.describe()[columnName]['max']
            descriptive.loc['IQR',columnName]=descriptive.loc['Q3:75%',columnName]-descriptive.loc['Q1:25%',columnName]
            descriptive.loc['1.5rule',columnName]=1.5*descriptive.loc['IQR',columnName]
            descriptive.loc['lesser',columnName]=descriptive.loc['Q1:25%',columnName]-descriptive.loc['1.5rule',columnName]
            descriptive.loc['greater',columnName]=descriptive.loc['Q3:75%',columnName]+descriptive.loc['1.5rule',columnName]
            descriptive.loc['min',columnName]=dataset[columnName].min()
            descriptive.loc['max',columnName]=dataset[columnName].max()
            descriptive.loc['skew',columnName]=dataset[columnName].skew()
            descriptive.loc['kurtosis',columnName]=dataset[columnName].kurtosis()
            descriptive.loc['std',columnName]=dataset[columnName].std()
            descriptive.loc['var',columnName]=dataset[columnName].var()
        return descriptive

    def findoutlier(descriptive,quan):
        lesser=[]
        greater=[]
        for colname in quan:
            if(descriptive.loc['min',colname]<descriptive.loc['lesser',colname]):
                lesser.append(colname)
            if(descriptive.loc['max',colname]>descriptive.loc['greater',colname]):
                greater.append(colname)
        return lesser,greater
        
    def repoutliers(dataset,descriptive,Lesser,Greater):
        for column in Lesser:
            dataset[column][dataset[column]<descriptive.loc['lesser',column]]=descriptive.loc['lesser',column]
        for column in Greater:
            dataset[column][dataset[column]>descriptive.loc['greater',column]]=descriptive.loc['greater',column]
        return dataset
    def pdf_proba(dataset,startrange,endrange):
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        ax=sns.distplot(dataset,kde=True,kde_kws={'color':'blue'},color='Green')
        plt.axvline(startrange,color='Red')
        plt.axvline(endrange,color='Red')
        sample=dataset
        sample_mean=sample.mean()
        sample_std=sample.std()
        print('Mean={:.3f},Satandard Deviation={:.3f}'.format(sample_mean,sample_std))
        dist=norm(sample_mean,sample_std)
        values=[value for value in range(startrange,endrange)]
        probabilities=[dist.pdf(values) for value in values]
        prob=sum(probabilities)
        print("The area range between({},{}):{}".format(startrange,endrange,sum(probabilities)))
        return prob
    def stdgraph(dataset):
        import seaborn as sns
        mean=dataset.mean()
        std=dataset.std()
        values=[i for i in dataset]
        z_score=[((j-mean)/std) for j in values]
        sns=sns.distplot(z_score,kde=True)
        sum(z_score)/len(z_score)
        return dataset