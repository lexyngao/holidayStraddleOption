import datetime
import jqdatasdk as jq
from jqdatasdk import *
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(filename='option_trading.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 设置聚宽登录（如果你尚未登录，需要先登录）
jq.auth('16621568150', 'Jq1234567')

# 获得交易日日期序列
end_date = datetime.date(2024, 10, 23)
start_date = datetime.date(2024, 9, 1)
date_list = list(jq.get_trade_days(start_date=start_date, end_date=end_date))

# 获得假期前一个工作日
open_date_list = []
close_date_list = []
for i in range(len(date_list) - 1):
    if (date_list[i + 1] - date_list[i]).days != 1:
        open_date_list.append(date_list[i])
        close_date_list.append(date_list[i + 1])


def get_50ETF_option(date, dfSignal=False):
    S0 = jq.get_price('510050.XSHG', start_date=date, end_date=date, fields=['close']).values[0][0]
    r = 0.03

    q = query(opt.OPT_DAILY_PREOPEN).filter(opt.OPT_DAILY_PREOPEN.date == date,
                                            opt.OPT_DAILY_PREOPEN.underlying_name == '50ETF')
    df = opt.run_query(q).loc[:, ['code', 'trading_code', 'exercise_price', 'exercise_date']]

    exercise_date_list = sorted(df['exercise_date'].unique())

    key_list = []
    Contract_dict = {}
    Price_dict = {}
    impVol_dict = {}

    for exercise_date in exercise_date_list:

        # 获得T型代码
        df1 = df[df['exercise_date'] == exercise_date]
        # 去除调整合约
        check = []
        for i in df1['trading_code']:
            x = True if i[11] == 'M' and i[6] == 'C' else False
            check.append(x)
        df_C = df1[check][['code', 'exercise_price']]
        df_C.index = df_C.exercise_price.values
        del df_C['exercise_price']
        df_C.columns = ['Call']
        df_C = df_C.sort_index()

        # 去除调整合约
        check = []
        for i in df1['trading_code']:
            x = True if i[11] == 'M' and i[6] == 'P' else False
            check.append(x)
        df_P = df1[check][['code', 'exercise_price']]
        df_P.index = df_P.exercise_price.values
        del df_P['exercise_price']
        df_P.columns = ['Put']
        df_P = df_P.sort_index()

        dfT = pd.concat([df_C, df_P], axis=1)
        exercise_date = datetime.datetime.strptime(str(exercise_date)[:10], '%Y-%m-%d')
        exercise_date = datetime.date(exercise_date.year, exercise_date.month, exercise_date.day)

        Contract_dict[exercise_date] = dfT

        # T型价格
        q = query(opt.OPT_DAILY_PRICE).filter(opt.OPT_DAILY_PRICE.date == date)
        df2 = opt.run_query(q).loc[:, ['code', 'close']]
        df2.index = df2['code'].values
        del df2['code']
        dfPrice = dfT.copy()
        dfPrice['Call'] = df2.loc[dfT.loc[:, 'Call'].values, :].values
        dfPrice['Put'] = df2.loc[dfT.loc[:, 'Put'].values, :].values
        dfPrice = dfPrice

        Price_dict[exercise_date] = dfPrice

        key_list.append(exercise_date)

    strike_price = list(Contract_dict[key_list[0]].index)
    atm_index = list(abs(np.round(S0 - strike_price, 3))).index(min(abs(np.round(S0 - strike_price, 3))))
    atm_K = strike_price[atm_index]

    if dfSignal:
        value_list = []
        for key, value in Contract_dict.items():
            value['exercise_date'] = key
            value_list.append(value)
        Contract_df = pd.concat(value_list).sort_values('exercise_date')

        value_list = []
        for key, value in Price_dict.items():
            value['exercise_date'] = key
            value_list.append(value)
        Price_df = pd.concat(value_list).sort_values('exercise_date')

        return Contract_df, Price_df, key_list, S0

    return Contract_dict, Price_dict, key_list, S0


def get_target_option(Contract_dict, Price_dict, key_list, S0, term=0, strategy=1):
    # Contract合约数据  Price价格数据  key_list是四个合约的到期日
    # S0当前标的价格 term远近合约选择 strategy策略

    expiry_date = key_list[term]
    Contract = Contract_dict[expiry_date]
    Price = Price_dict[expiry_date]
    strike_price = list(Contract.index)

    # 选择跨式期权位置

    check = min(abs(np.round(S0 - strike_price, 3)))
    if check < 0.01:
        strategy = 0
    else:
        strategy = strategy

    if strategy == 0:
        atm_index = list(abs(np.round(S0 - strike_price, 3))).index(min(abs(np.round(S0 - strike_price, 3))))
        atm_k = strike_price[atm_index]
        result = pd.DataFrame(index=['Strike', 'ticker', 'Price'], columns=['Call', 'Put'])
        result.loc['Strike', 'Call'] = atm_k
        result.loc['Strike', 'Put'] = atm_k
        result.loc['ticker', 'Call'] = Contract.loc[atm_k, 'Call']
        result.loc['ticker', 'Put'] = Contract.loc[atm_k, 'Put']
        result.loc['Price', 'Call'] = Price.loc[atm_k, 'Call']
        result.loc['Price', 'Put'] = Price.loc[atm_k, 'Put']

    if strategy == 1:

        for i in range(len(strike_price) - 1):
            if strike_price[i] <= S0 and strike_price[i + 1] > S0:
                k1 = strike_price[i]
                k2 = strike_price[i + 1]

        result = pd.DataFrame(index=['Strike', 'ticker', 'Price'], columns=['Call', 'Put'])
        result.loc['Strike', 'Call'] = k2
        result.loc['Strike', 'Put'] = k1
        result.loc['ticker', 'Call'] = Contract.loc[k2, 'Call']
        result.loc['ticker', 'Put'] = Contract.loc[k1, 'Put']
        result.loc['Price', 'Call'] = Price.loc[k2, 'Call']
        result.loc['Price', 'Put'] = Price.loc[k1, 'Put']

    if strategy == 2:

        for i in range(len(strike_price) - 1):
            if strike_price[i] <= S0 and strike_price[i + 1] > S0:
                k1 = strike_price[i - 1]
                k2 = strike_price[i + 2]

        result = pd.DataFrame(index=['Strike', 'ticker', 'Price'], columns=['Call', 'Put'])
        result.loc['Strike', 'Call'] = k2
        result.loc['Strike', 'Put'] = k1
        result.loc['ticker', 'Call'] = Contract.loc[k2, 'Call']
        result.loc['ticker', 'Put'] = Contract.loc[k1, 'Put']
        result.loc['Price', 'Call'] = Price.loc[k2, 'Call']
        result.loc['Price', 'Put'] = Price.loc[k1, 'Put']

    return result


def get_50ETF_price_byday(date, Code):
    q = query(opt.OPT_DAILY_PRICE).filter(opt.OPT_DAILY_PRICE.date == date, opt.OPT_DAILY_PRICE.code == Code)
    df = opt.run_query(q).loc[:, ['code', 'close']]
    price = df.loc[0, 'close']
    return price


N = len(open_date_list)

columns = ['t', 'St', 'CK', 'PK', 'Callcode', 'Putcode', 'Callt', 'Putt', 'T', 'ST', 'CallT', 'PutT']
df = pd.DataFrame(index=list(range(N)), columns=columns)

for i in range(N):
    open_date = open_date_list[i]
    Contract_dict, Price_dict, key_list, S0 = get_50ETF_option(open_date, dfSignal=False)
    table = get_target_option(Contract_dict, Price_dict, key_list, S0, 1, 1)

    df.loc[i, 't'] = open_date
    df.loc[i, 'St'] = S0
    df.loc[i, 'CK'] = table.loc['Strike', 'Call']
    df.loc[i, 'PK'] = table.loc['Strike', 'Put']
    Callcode = table.loc['ticker', 'Call']
    Putcode = table.loc['ticker', 'Put']
    df.loc[i, 'Callcode'] = Callcode
    df.loc[i, 'Putcode'] = Putcode
    df.loc[i, 'Callt'] = table.loc['Price', 'Call']
    df.loc[i, 'Putt'] = table.loc['Price', 'Put']

    # 记录买入日志
    logging.info(f"交易日期: {open_date}, 交易合约: {Callcode}(Call), 交易份数: 1, 交易价格: {df.loc[i, 'Callt']}")
    logging.info(f"交易日期: {open_date}, 交易合约: {Putcode}(Put), 交易份数: 1, 交易价格: {df.loc[i, 'Putt']}")

    close_date = close_date_list[i]

    df.loc[i, 'T'] = close_date
    df.loc[i, 'ST'] = get_price('510050.XSHG', start_date=close_date, end_date=close_date, fields=['close']).values[0][
        0]
    df.loc[i, 'CallT'] = get_50ETF_price_byday(close_date, Callcode)
    df.loc[i, 'PutT'] = get_50ETF_price_byday(close_date, Putcode)

    # 记录卖出日志
    logging.info(f"交易日期: {close_date}, 交易合约: {Callcode}(Call), 交易份数: 1, 交易价格: {df.loc[i, 'CallT']}")
    logging.info(f"交易日期: {close_date}, 交易合约: {Putcode}(Put), 交易份数: 1, 交易价格: {df.loc[i, 'PutT']}")

df['ret'] = (df['CallT'] + df['PutT']) / (df['Callt'] + df['Putt']) - 1
df['t'] = pd.to_datetime(df['t'])

# 绘制单期收益的柱状图
plt.figure(figsize=(12, 6))
plt.bar(df['t'], df['ret'], color='#003778')  # 使用plt.bar来绘制柱状图
plt.title('Single Period Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Return')
plt.xticks(df['t'], df['t'].dt.strftime('%Y-%m-%d'), rotation=45)  # 设置横坐标显示为日期，并旋转45度以便查看
plt.grid(True)
plt.legend()
plt.show()

# 计算胜率
win_rate = (df['ret'] > 0).mean()

# 计算平均正收益和平均负收益
average_positive_return = df[df['ret'] > 0]['ret'].mean()
average_negative_return = df[df['ret'] < 0]['ret'].mean()

# 计算赔率
if average_negative_return != 0:
    odds = abs(average_positive_return / average_negative_return)
else:
    odds = float('inf')  # 如果没有负收益，赔率无限大

print(f"胜率: {win_rate*100:.2f}%")
print(f"赔率: {odds:.2f}")
