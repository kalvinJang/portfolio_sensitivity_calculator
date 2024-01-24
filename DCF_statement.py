import pandas as pd
def FCF(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/income-statement/{}?limit=120&apikey={}'.format(code, api_key)
    pd_is = pd.read_json(url)
    pd_fcf = pd_is[['date', 'ebitda', 'netIncome', 'interestExpense', 'interestIncome', 'depreciationAndAmortization', 'operatingIncome', 'operatingExpenses', 'researchAndDevelopmentExpenses', 'otherExpenses']]
    netoperating = (pd_ebitda['operatingExpenses'] - pd_ebitda['operatingIncome']).diff(-1).fillna(0)
    pd_fcf['netoperating'] = netoperating
    pd_fcf['fcf'] = pd_ebitda['netIncome'] + pd_ebitda['interestExpense'] - pd_ebitda['interestIncome'] +pd_ebitda['depreciationAndAmortization'] - pd_ebitda['netoperating'] - pd_ebitda['researchAndDevelopmentExpenses'] + pd_ebitda['otherExpenses']
    return  pd_fcf[['date', 'fcf']]


def WACC(code):
    api_key = '*************************'
    url = 'https://financialmodelingprep.com/api/v3/income-statement/{}?limit=120&apikey={}'.format(code, api_key)
    
