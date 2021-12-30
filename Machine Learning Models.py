import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
from finrl.apps import config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
##########################################################################################################################
df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2021-09-02',
                     ticker_list = config.DOW_30_TICKER).fetch_data()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)
# add covariance matrix as states
df = df.sort_values(['date', 'tic'], ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i - lookback:i, :]
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)

    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date', 'tic']).reset_index(drop=True)
train = data_split(df, '2009-01-01','2020-06-30')
stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-1

}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()


trade = data_split(df,'2020-07-01', '2021-09-02')
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)

unique_tic = trade.tic.unique()
unique_trade_date = trade.date.unique()
######################################################################################################################
def prepare_data(trainData):
  train_date = sorted(set(trainData.date.values))
  X = []
  for i in range(0, len(train_date) - 1):
    d = train_date[i]
    d_next = train_date[i+1]
    y = train.loc[train['date'] == d_next].return_list.iloc[0].loc[d_next].reset_index()
    y.columns = ['tic', 'return']
    x = train.loc[train['date'] == d][['tic','macd','rsi_30','cci_30','dx_30']]
    train_piece = pd.merge(x, y, on = 'tic')
    train_piece['date'] = [d] * len(train_piece)
    X += [train_piece]
  trainDataML = pd.concat(X)
  X = trainDataML[['macd', 'rsi_30', 'cci_30', 'dx_30']].values
  Y = trainDataML[['return']].values

  return X, Y

train_X, train_Y = prepare_data(train)
rf_model = RandomForestRegressor(max_depth = 35,  min_samples_split = 10, random_state = 0).fit(train_X, train_Y.reshape(-1))
dt_model = DecisionTreeRegressor(random_state = 0, max_depth=35, min_samples_split = 10 ).fit(train_X, train_Y.reshape(-1))
svm_model =  SVR(epsilon=0.14).fit(train_X, train_Y.reshape(-1))
lr_model = LinearRegression().fit(train_X, train_Y)


########################################################################################################################
def output_predict(model, reference_model=False):
  meta_coefficient = {"date": [], "weights": []}

  portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
  initial_capital = 1000000
  portfolio.loc[0, unique_trade_date[0]] = initial_capital

  for i in range(len(unique_trade_date) - 1):

    current_date = unique_trade_date[i]
    next_date = unique_trade_date[i + 1]
    df_current = df[df.date == current_date].reset_index(drop=True)
    tics = df_current['tic'].values
    features = df_current[['macd', 'rsi_30', 'cci_30', 'dx_30']].values
    df_next = df[df.date == next_date].reset_index(drop=True)
    if not reference_model:
      predicted_y = model.predict(features)
      mu = predicted_y
      Sigma = risk_models.sample_cov(df_current.return_list[0], returns_data=True)
    else:
      mu = df_next.return_list[0].loc[next_date].values
      Sigma = risk_models.sample_cov(df_next.return_list[0], returns_data=True)
    predicted_y_df = pd.DataFrame({"tic": tics.reshape(-1, ), "predicted_y": mu.reshape(-1, )})
    min_weight, max_weight = 0, 1
    ef = EfficientFrontier(mu, Sigma)
    weights = ef.nonconvex_objective(
      objective_functions.sharpe_ratio,
      objective_args=(ef.expected_returns, ef.cov_matrix),
      weights_sum_to_one=True,
      constraints=[
        {"type": "ineq", "fun": lambda w: w - min_weight},  # greater than min_weight
        {"type": "ineq", "fun": lambda w: max_weight - w},  # less than max_weight
      ],
    )

    weight_df = {"tic": [], "weight": []}
    meta_coefficient["date"] += [current_date]
    # it = 0
    for item in weights:
      weight_df['tic'] += [item]
      weight_df['weight'] += [weights[item]]

    weight_df = pd.DataFrame(weight_df).merge(predicted_y_df, on=['tic'])
    meta_coefficient["weights"] += [weight_df]
    cap = portfolio.iloc[0, i]
    # current cash invested for each stock
    current_cash = [element * cap for element in list(weights.values())]
    # current held shares
    current_shares = list(np.array(current_cash) / np.array(df_current.close))
    # next time period price
    next_price = np.array(df_next.close)
    portfolio.iloc[0, i + 1] = np.dot(current_shares, next_price)

  portfolio = portfolio.T
  portfolio.columns = ['account_value']
  portfolio = portfolio.reset_index()
  portfolio.columns = ['date', 'account_value']
  stats = backtest_stats(portfolio, value_col_name='account_value')
  portfolio_cumprod = (portfolio.account_value.pct_change() + 1).cumprod() - 1

  return portfolio, stats, portfolio_cumprod, pd.DataFrame(meta_coefficient)


lr_portfolio, lr_stats, lr_cumprod, lr_weights = output_predict(lr_model)
dt_portfolio, dt_stats, dt_cumprod, dt_weights = output_predict(dt_model)
svm_portfolio, svm_stats, svm_cumprod, svm_weights = output_predict(svm_model)
rf_portfolio, rf_stats, rf_cumprod, rf_weights = output_predict(rf_model)
reference_portfolio, reference_stats, reference_cumprod, reference_weights = output_predict(None, True)