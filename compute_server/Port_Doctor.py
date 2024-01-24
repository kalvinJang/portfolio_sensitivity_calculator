import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from itertools import combinations
from sklearn.linear_model import RANSACRegressor 



def forward_stepwise_selection(df):
    ## 전진 단계별 선택법
    variables = df.columns[:-1].tolist() ## 설명 변수 리스트
    y = df['Y'] ## 반응 변수
    selected_variables = [] ## 선택된 변수들
    sl_enter = 0.1  
    sl_remove = 0.1  

    sv_per_step = [] ## 각 스텝별로 선택된 변수들
    adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
    bic_per_step = [] ## 각 스텝별 bic
    steps = [] ## 스텝
    step = 0
    i= 0 ## 무한루프 방지용
    while len(variables) > 0:
        if i >= 3*len(variables) :
            break
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder) ## 변수의 p-value
        ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서 
        ## 선형 모형을 적합한다.
        for col in remainder: 
            X = df[selected_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y,X).fit()
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        i+=1
        
        if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
            selected_variables.append(pval.idxmin())
            ## 선택된 변수들에대해서 어떤 변수를 제거할지 고르기
            while len(selected_variables) > 0:
                selected_X = df[selected_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y,selected_X).fit().pvalues[1:] ## 절편항의 p-value는 빼기
                max_pval = selected_pval.max()
                if max_pval >= sl_remove: ## 최대 p-value값이 기준값보다 크거나 같으면 제외
                    remove_variable = selected_pval.idxmax()
                    selected_variables.remove(remove_variable)
                else:
                    break

            step += 1
            steps.append(step)
            model = sm.OLS(y,sm.add_constant(df[selected_variables])).fit()
            pvalue = model.pvalues
            adjusted_r_squared.append(model.rsquared_adj)
            sv_per_step.append(selected_variables.copy())
            bic_per_step.append(model.bic)
        else:
            break
    return selected_variables, adjusted_r_squared, pvalue

def del_duplicate(list):
    new_list = []
    for v in list:
        if v not in new_list:
            new_list.append(v)
    return new_list

def SOLVE_MULTICOL(KOR, KOR_return, KOR_monthly, term='M', start = '2010-01-01', end='2022-10-01'):

    KOR_monthly.index = pd.to_datetime(KOR_monthly.index)  #물가, 실업률
    KOR_monthly = KOR_monthly.resample('M').sum()
    
#     stock = KOR_return.loc[KOR.index[0]:,'035760']
#     stock.index = pd.to_datetime(stock.index)
#     stock = stock.resample(term).mean()
    stock = KOR_return
    
    left = KOR.loc[:str(stock.index[-1]), :]
    left.index = pd.to_datetime(left.index)
    left = left.resample(term).sum()
    
    ols_table = pd.merge(left, stock, left_on=left.index.astype(str), right_on=stock.index.astype(str))
    ols_table.index = ols_table['key_0']
    ols_table.drop('key_0', axis=1, inplace=True)
    ols_table_resample = ols_table
    
    if term=='M' or 'Q':
        ols_table_resample = pd.merge(KOR_monthly, ols_table, left_on=KOR_monthly.index.astype(str), right_on=ols_table.index.astype(str))
        ols_table_resample.index = ols_table_resample['key_0']
        ols_table_resample.drop('key_0', axis=1, inplace=True)
        ols_table_resample.index = pd.to_datetime(ols_table_resample.index)

        if term == 'Q':
            ols_table_resample = ols_table_resample.resample(term).sum()

      ### 여기서는 Y 안 씀. 다만 위에서 정한 기간은 사용함.
    VIF_table = ols_table_resample.copy().iloc[:,:-1]
    VIF_table = VIF_table.loc[start : end, :]              #테스트기간
    A_list = []
    final_candidate = []
    delete_list = []
    
    for i in VIF_table.columns:
        candidate = []
        inter_candidate = []
        len_candidate = 100
        X = VIF_table.drop(i, axis=1).columns

        for j in range(1,len(X)//2+1):    #여기에 for문 걸어서 X들의 모든 조합(조합의 최대길이는 x//2로 제한)이 가능하도록
            comb = list(combinations(X, j))    ## tuple들이 list안에 있는 구조
            for k in range(len(comb)):
                k_th = list(comb[k])
                regressor = '+'.join([x for x in k_th])
                re = ols('{} ~ {}'.format(i, regressor), VIF_table).fit()
                Rsquare = re.rsquared_adj
                if Rsquare >=0.8:
                    candidate.append(k_th)
                else:
                    pass #다음 조합으로 이동
                
        #Rsquare 0.8이상인 조합을 다 찾고난 후 길이가 가장 짧은 애들을 원소로 하는 final candidate_list를 만듦
        for sole in candidate: #sole is list type
            if len(sole) <=len_candidate:
                inter_candidate.append(sole)
                len_candidate = len(sole)
            else:
                pass
        for sole_2 in inter_candidate:
            sole_2.insert(0, i)
            if len(sole_2) <= len_candidate +1:
                final_candidate.append(sole_2)
        ### 여기까지의 결과값은 길이가 len_candidate로 가장 작아진 X들의 list ( i도 포함됨)들이 모여있는 final_candidate [ [X1,X2,X4], [X1,X4,X5] , [X2,X4,X5] ...... ]
        
    #final_candidate에 있는 것 중 똑같은 list 없애주기
    final_candidate = del_duplicate(final_candidate)

#     ### Y에 해당하는 맨 앞 element를 뽑고 First_Y list로 놓고, 이 first_Y의 원소들로만 나머지 final_candidate의 애들이 구성되어있으면 그 Y는 A_list로 이동. FIrst_Y와 겹치지 않으면 delete lsit로 이동

    Y_to_delete_list = []  ## 다중공선성이 있을 것으로 의심되는 Y들
    final_candidates_for_loop = final_candidate.copy()
    for F in final_candidates_for_loop:
        Y_to_delete = F.pop(0)   #그럼 [X1,X2,X4]에서 F=[X2,X4], Y_to_delete ='X1' 으로 되어있음
        F_prior = F.copy()
        Y_to_delete_list.append(Y_to_delete)
        ##만약 F_prior의 모든 원소가 Y_to_delete_list에 포함된다면 A_list로 넘기기
        if set(F_prior).intersection(set(Y_to_delete_list)) == set(F_prior):
            A_list.append(Y_to_delete)
        ##elif F_prior의 모든 원소가 Y_to_delete_list에 포함되지 않는다면 delete_list로 넘기기
        elif set(F_prior).intersection(set(Y_to_delete_list)) == set():
            delete_list.append(Y_to_delete)
    delete_list = del_duplicate(delete_list)
    A_list = del_duplicate(A_list)
    A_list = list(set(A_list).difference(set(delete_list)))
    
    return A_list, delete_list


def CALCULATION_SENSITIVITY(KOR, KOR_return, KOR_monthly, A_list, delete_list, stock_list, weight, DO_MULTICOL=False, ransac=False, term='M', start = '2010-01-01', end='2022-01-01'):    
    global ols_table_resample
    
    stock_sensitivity = pd.Series()
    KOR_monthly.index = pd.to_datetime(KOR_monthly.index)
    KOR_monthly = KOR_monthly.loc[start : end, :]
    
    stock = KOR_return.loc[KOR.index[0]: , stock_list]
    stock.index = pd.to_datetime(stock.index)
    stock = stock.loc[start : end, :]
    

    left = KOR.loc[:str(stock.index[-1]), :]
    left.index = pd.to_datetime(left.index)
    left = left.loc[start : end, :]

    result_var = [x for x in KOR_monthly.columns] + [x for x in left.columns]   ##최종 outcome에는 모든 변수가 다 들어있어야하니까!ㄴ

    ## 3개월 또는 3분기 연속 NaN인 변수 있으면 이 period에 대해서는 빼고 돌리기.
    left_pre = left.copy()
    key=[]
    for col in left_pre.columns[:-1]:
        if left_pre[col].isna().sum() != 0:
            for j in range(1,len(np.where(left_pre[col].isna())[0])):
                key.append(np.where(left_pre[col].isna())[0][j-1] - np.where(left_pre[col].isna())[0][j])
            for k in range(len(key)-1):
                if key[k]==-1 & key[k+1]==-1:   #독립변수 : 해당 기간에 3번 연속 nan있는 column은 col자체를 지우기
                    left.drop(columns=[col], inplace=True)
                    break
                else:
                    pass
        else:
            pass
        if col in left.columns:
            left[col].fillna(method='ffill', inplace=True)  #먼저 ffill하고
            left[col].fillna(method='bfill', inplace=True)  #맨 처음 row에 있는 애들은 bfill하기          
    
    KOR_monthly = KOR_monthly.resample('M').sum()
    stock = stock.resample(term).mean()
    left = left.resample(term).sum()
    
    ols_table = pd.merge(left, stock, left_on=left.index.astype(str), right_on=stock.index.astype(str))
    ols_table.index = ols_table['key_0']
    ols_table.drop('key_0', axis=1, inplace=True)
    ols_table_resample = ols_table

    if term=='M' or 'Q':
        ols_table_resample = pd.merge(KOR_monthly, ols_table, left_on=KOR_monthly.index.astype(str), right_on=ols_table.index.astype(str))
        ols_table_resample.index = ols_table_resample['key_0']
        ols_table_resample.drop('key_0', axis=1, inplace=True)
        ols_table_resample.index = pd.to_datetime(ols_table_resample.index)

        if term == 'Q':
            ols_table_resample = ols_table_resample.resample(term).sum()    
    
    var = [x for x in KOR_monthly.columns] + [x for x in left.columns]    

    if DO_MULTICOL:
        A_list, delete_list = SOLVE_MULTICOL(KOR, KOR_return, KOR_monthly, term='M', start =start, end=end)
    
###########################################################################################    
# 포트폴리오 수익률 stock_list ols_table_resample에 추가해야함

    weight_sum = np.sum(weight)
    weight = pd.Series(weight).map(lambda x: x/weight_sum).tolist()
    ols_table_resample['Portfolio'] = (weight * ols_table_resample[stock_list]).sum(1)
    stock_list.append('Portfolio')

    for code in stock_list:
        try:
            ols_table_stock = ols_table_resample[var+[code]]
            ols_table_stock.columns = [x for x in ols_table_stock.columns[:-1]] +['Y']
            ols_table_training = ols_table_stock.copy()
            
            if len(A_list) != 0:
                select = ols_table_training[A_list+['Y']]
                min_pvalue=1
                for i, x in enumerate(A_list):
                    temp =ols('Y ~ {}'.format(x), select).fit()
                    if temp.pvalues[1] < min_pvalue:
                        min_pvalue = temp.pvalues[1]
                        min_index = i
                        min_X = x
                A_list.pop(i)   #pvalue가 가장 작은 애를 A_list에서 삭제
                delete_list.extend(A_list)  #A_list 안에 있는 모든 원소를 delete_list에 추가

            ### 종목마다 delete_list 애들을 ols_table_training에서 없애기
            new_column = list(set([x for x in ols_table_training.columns[:-1]]).difference(set(delete_list)))
            new_data_training = ols_table_training[new_column + ['Y']]
            
            try:
                selected_variables, adjusted_r_squared, selection_pvalue = forward_stepwise_selection(new_data_training)
                if len(selected_variables)!=0:
                    if ransac:
                        reg_est = []  
                        for k in range(10):
                            reg = RANSACRegressor(random_state=k, min_samples=0.85, residual_threshold=0.011, max_trials=500, ).fit(new_data_training[selected_variables], new_data_training['Y'])
                            reg_est.append([reg.estimator_.intercept_] + reg.estimator_.coef_.tolist())
                        Ransac_coef = np.array(reg_est).mean(0).tolist()
                        stock_sensitivity_sub = pd.Series(Ransac_coef, index = ['Intercept']+selected_variables, name=code)
                    else:
                        variable = '+'.join(selected_variables)
                        model = ols('Y ~ {}'.format(variable) , new_data_training).fit()  
                        stock_sensitivity_sub = model.params #pd.Series
                        stock_sensitivity_sub.name = code
                else:
                    stock_sensitivity_sub = pd.Series(None, name=code)
                    pass
            except:
                stock_sensitivity_sub = pd.Series(None, name=code)
                pass
        except AttributeError:
            stock_sensitivity_sub = pd.Series(None, name=code)
            pass
        stock_sensitivity = pd.concat([stock_sensitivity, stock_sensitivity_sub], axis=1)
    
    surplus = list(set(result_var).difference(set(stock_sensitivity.index)))
    stock_sensitivity = stock_sensitivity.append(pd.DataFrame(None, index=surplus)).fillna(0)
    stock_sensitivity_basic = stock_sensitivity.iloc[:, 1:]   ## 포트폴리오 변수별 관련도 및 포트폴리오 구성 종목별 관련도
    
    Macro = stock_sensitivity.index[1:].tolist()
    for macro in Macro:
        globals()[str(macro)] = dict()
        pos_ = np.where(stock_sensitivity_basic.loc[macro][:-1]>0)[0].tolist() ##해당 macro var에 대해 양수인 종목의 위치
        neg_ = np.where(stock_sensitivity_basic.loc[macro][:-1]<0)[0].tolist()
        if len(pos_)>0:
            globals()[str(macro)]['확대'] = pd.Series(stock_list)[pos_].tolist()
        elif len(pos_) == 0:
            globals()[str(macro)]['확대'] = []
        if len(neg_)>0:
            globals()[str(macro)]['축소'] = pd.Series(stock_list)[neg_].tolist()
        elif len(neg_) ==0:
            globals()[str(macro)]['축소'] = []
        
    return stock_sensitivity_basic, [globals()[str(macro)] for macro in Macro], Macro