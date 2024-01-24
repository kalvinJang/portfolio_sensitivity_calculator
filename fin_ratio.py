import pandas as pd
def fmp_fin_ratio(code):
    api_key ='*************************'
    url = "https://financialmodelingprep.com/api/v3/ratios/{}?apikey={}".format(code, api_key)
    pd_ratio = pd.read_json(url)
    return pd_ratio, pd_ratio.columns
#Receivables Turnover : 매출채권회전율
#payablesTurnover : 매입채무회전율
#asset Turnover : 총자산회전율
#inventoryTurnover : 재고자산회전율
#'priceBookValueRatio', 'priceToBookRatio','priceToSalesRatio', 'priceEarningsRatio', 'priceToFreeCashFlowsRatio',
# 'priceToOperatingCashFlowsRatio', 'priceCashFlowRatio',
# 'priceEarningsToGrowthRatio', 'priceSalesRatio'
# enterpriseValueMultiple : EV / EBITDA
# debtRatio 부채비율, debtEquityRatio : 부채자본비율
# 'effectiveTaxRate' 실효세율
# 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed

