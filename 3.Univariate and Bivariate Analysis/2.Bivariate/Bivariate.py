class bivariate:
    def quanqual(dataset):
        quan=[]
        qual=[]
        for columnName in dataset.columns:
            if (dataset[columnName].dtype=='O'):
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan,qual
        
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    def calc_vif(x):
        vif=pd.DataFrame()
        vif['Varaince']=x.columns
        vif['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
        return vif
        
    def indep_ttest(dataset,group_col,group1,group2,value_col):
        from scipy.stats import ttest_ind
        Male=dataset[dataset[group_col]==group1][value_col]
        Female=dataset[dataset[group_col]==group2][value_col]
        ttest=ttest_ind(Male,Female)
        pvalue=ttest.pvalue
        print('P_Value:',pvalue)
        if(pvalue<0.05):
            print("Reject null hypothesis:There is a significant relationship between given varables")
        else:
            print("Fail to Reject null hypothesis: There is no significant relationship between given varables")
        return ttest
        
    def dep_ttest(dataset,group_col,group,value1,value2):  
        from scipy.stats import ttest_rel
        male=dataset[dataset[group_col]==group][value1]
        male1=dataset[dataset[group_col]==group][value2]
        ttest=ttest_rel(male,male1)
        pvalue=ttest.pvalue
        print("P_Value:",pvalue)
        if(pvalue<0.05):
            print("Reject null hypothesis:There is a significant relationship between given varables")
        else:
            print("Fail to Reject null hypothesis: There is no significant relationship between given varables")
        return ttest
        
    def anova_oneway(dataset,var1,var2,var3):
        from scipy.stats import stats
        one_way=stats.f_oneway(dataset[var1],dataset[var2],dataset[var3])
        pvalue=one_way.pvalue
        print('one_way_pvalue:',pvalue)
        if(pvalue<0.05):
            print("Reject null hypothesis:There is a significant relationship between given varables")
        else:
            print("Fail to Reject null hypothesis: There is no significant relationship between given varables")
        return one_way
        
    def anova_two_way(dataset,numer,cate1,cate2):
        import statsmodels.api as sm   #creating the table    
        from statsmodels.formula.api import ols   #creating the model
        model=ols(f'{numer}~C({cate1})+C({cate2})+C({cate1}):C({cate2})',data=dataset).fit()
        anova_table = sm.stats.anova_lm(model, type=2)
        return anova_table