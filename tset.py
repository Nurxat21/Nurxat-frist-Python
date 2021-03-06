# Basic Packages
import numpy as np
import pandas as pd
import yfinance as yf


# Advance Packages
import pytz
import exchange_calendars as tc
from stockstats import StockDataFrame as Sdf
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv





env = StockTradingEnv

drl_lib='elegantrl'
TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'
time_interval = "1D"
TECHNICAL_INDICATORS_LIST = ['macd',
                             'boll_ub',
                             'boll_lb',
                             'rsi_30',
                             'dx_30',
                             'close_30_sma',
                             'close_60_sma']

ERL_PARAMS = {"learning_rate": 3e-5,"batch_size": 2048,"gamma":  0.985,
              "seed":312,"net_dimension":512, "target_step":5000, "eval_gap":60}
DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V"
]

"""
DownLoad Training Data by Yahoo Finance
"""
# Download data
data_df = pd.DataFrame()
for tic in DOW_30_TICKER:
    temp_df = yf.download(tic, start=TRAIN_START_DATE, end=TRAIN_END_DATE)
    temp_df["tic"] = tic
    data_df = data_df.append(temp_df)
try:
    # convert the column names to standardized names
    data_df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjcp",
        "volume",
        "tic",
    ]
except NotImplementedError:
    print("the features are not supported currently")
# create day of the week column (monday = 0)
data_df["day"] = data_df["date"].dt.dayofweek
# convert date to standard string format, easy to filter
data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
# drop missing data
data_df = data_df.dropna()
data_df = data_df.reset_index(drop=True)
print("Shape of DataFrame: ", data_df.shape)
# print("Display DataFrame: ", data_df.head())

data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
# Clean data
df = data_df.copy()
df = df.rename(columns={"date": "time"})
time_interval = time_interval
# get ticker list
tic_list = np.unique(df.tic.values)
#######################################################################
def get_trading_days(self, start, end):
    nyse = tc.get_calendar("NYSE")
    df = nyse.sessions_in_range(
        pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
    )
    trading_days = []
    for day in df:
        trading_days.append(str(day)[:10])
########################################################################
# get complete time index
trading_days = get_trading_days(start=TRAIN_START_DATE, end=TRAIN_END_DATE)
if time_interval == "1D":
    times = trading_days
elif time_interval == "1Min":
    times = []
    for day in trading_days:
        NY = "America/New_York"
        current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
        for i in range(390):
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)
else:
    raise ValueError(
        "Data clean at given time interval is not supported for YahooFinance data."
    )
# fill NaN data
new_df = pd.DataFrame()
for tic in tic_list:
    print(("Clean data for ") + tic)
    # create empty DataFrame using complete time index
    tmp_df = pd.DataFrame(
        columns=["open", "high", "low", "close", "adjcp", "volume"], index=times
    )
    # get data for current ticker
    tic_df = df[df.tic == tic]
    # fill empty DataFrame using orginal data
    for i in range(tic_df.shape[0]):
        tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
            ["open", "high", "low", "close", "adjcp", "volume"]
        ]

    # if close on start date is NaN, fill data with first valid close
    # and set volume to 0.
    if str(tmp_df.iloc[0]["close"]) == "nan":
        print("NaN data on start date, fill using first valid data.")
        for i in range(tmp_df.shape[0]):
            if str(tmp_df.iloc[i]["close"]) != "nan":
                first_valid_close = tmp_df.iloc[i]["close"]
                first_valid_adjclose = tmp_df.iloc[i]["adjcp"]

        tmp_df.iloc[0] = [
            first_valid_close,
            first_valid_close,
            first_valid_close,
            first_valid_close,
            first_valid_adjclose,
            0.0,
        ]

    # fill NaN data with previous close and set volume to 0.
    for i in range(tmp_df.shape[0]):
        if str(tmp_df.iloc[i]["close"]) == "nan":
            previous_close = tmp_df.iloc[i - 1]["close"]
            previous_adjcp = tmp_df.iloc[i - 1]["adjcp"]
            if str(previous_close) == "nan":
                raise ValueError
            tmp_df.iloc[i] = [
                previous_close,
                previous_close,
                previous_close,
                previous_close,
                previous_adjcp,
                0.0,
            ]

    # merge single ticker data to new DataFrame
    tmp_df = tmp_df.astype(float)
    tmp_df["tic"] = tic
    new_df = new_df.append(tmp_df)

    print(("Data clean for ") + tic + (" is finished."))

# reset index and rename columns
new_df = new_df.reset_index()
new_df = new_df.rename(columns={"index": "time"})

print("Data clean all finished!")
df = new_df.copy()
df = df.sort_values(by=["tic", "time"])
stock = Sdf.retype(df.copy())
unique_ticker = stock.tic.unique()

for indicator in TECHNICAL_INDICATORS_LIST:
    indicator_df = pd.DataFrame()
    for i in range(len(unique_ticker)):
        try:
            temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
            temp_indicator = pd.DataFrame(temp_indicator)
            temp_indicator["tic"] = unique_ticker[i]
            temp_indicator["time"] = df[df.tic == unique_ticker[i]][
                "time"
            ].to_list()
            indicator_df = indicator_df.append(
                temp_indicator, ignore_index=True
            )
        except Exception as e:
            print(e)
    df = df.merge(
        indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
    )
df = df.sort_values(by=["time", "tic"])
##################################################################################################################
def add_vix(self, data):
    """
    add vix from yahoo finance
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = data.copy()
    df_vix = self.download_data(
        start_date=df.time.min(),
        end_date=df.time.max(),
        ticker_list=["^VIX"],
        time_interval=self.time_interval,
    )
    df_vix = self.clean_data(df_vix)
    vix = df_vix[["time", "adjcp"]]
    vix.columns = ["time", "vix"]

    df = df.merge(vix, on="time")
    df = df.sort_values(["time", "tic"]).reset_index(drop=True)
    return df
#######################################################################################################################

if_vix=True
if if_vix:
    unique_ticker = df.tic.unique()
    print(unique_ticker)
    if_first_time = True
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][["adjcp"]].values
            # price_ary = df[df.tic==tic]['close'].values
            tech_array = df[df.tic == tic][TECHNICAL_INDICATORS_LIST].values
            if if_vix:
                turbulence_array = df[df.tic == tic]["vix"].values
            else:
                turbulence_array = df[df.tic == tic]["turbulence"].values
            if_first_time = False
        else:
            price_array = np.hstack(
                [price_array, df[df.tic == tic][["adjcp"]].values]
            )
            tech_array = np.hstack(
                [tech_array, df[df.tic == tic][TECHNICAL_INDICATORS_LIST].values]
            )
    assert price_array.shape[0] == tech_array.shape[0]
    assert tech_array.shape[0] == turbulence_array.shape[0]
    print("Successfully transformed into array")

    unique_ticker = df.tic.unique()
    print(unique_ticker)
    if_first_time = True
    for tic in unique_ticker:
        if if_first_time:
            price_array = df[df.tic == tic][["adjcp"]].values
            # price_ary = df[df.tic==tic]['close'].values
            tech_array = df[df.tic == tic][TECHNICAL_INDICATORS_LIST].values
            if if_vix:
                turbulence_array = df[df.tic == tic]["vix"].values
            else:
                turbulence_array = df[df.tic == tic]["turbulence"].values
            if_first_time = False
        else:
            price_array = np.hstack(
                [price_array, df[df.tic == tic][["adjcp"]].values]
            )
            tech_array = np.hstack(
                [tech_array, df[df.tic == tic][TECHNICAL_INDICATORS_LIST].values]
            )
    assert price_array.shape[0] == tech_array.shape[0]
    assert tech_array.shape[0] == turbulence_array.shape[0]
    print("Successfully transformed into array")
# fill nan and inf values with 0 for technical indicators
tech_nan_positions = np.isnan(tech_array)
tech_array[tech_nan_positions] = 0
tech_inf_positions = np.isinf(tech_array)
tech_array[tech_inf_positions] = 0

env_config = {'price_array': price_array,
              'tech_array': tech_array,
              'turbulence_array': turbulence_array,
              'if_train': True}
env_instance = env(config=env_config)
model_name='ppo'
###########################################################################################################################
MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "a2c": AgentA2C}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]
class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        env = self.env(config=env_config)
        env.env_num = 1
        agent = MODELS[model_name]()
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Arguments(env, agent)
        if model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = MODELS[model_name]()
        environment.env_num = 1
        args = Arguments(env=environment, agent=model)
        if model_name in OFF_POLICY_MODELS:
            args.if_off_policy = True
        else:
            args.if_off_policy = False
        args.agent = model
        args.env = environment
        # args.agent.if_use_cri_target = True  ##Not needed for test

        # load agent
        try:
            state_dim = environment.state_dim
            action_dim = environment.action_dim

            agent = args.agent
            net_dim = net_dimension

            agent.init(net_dim, state_dim, action_dim)
            agent.save_or_load_agent(cwd=cwd, if_save=False)
            act = agent.act
            device = agent.device

        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = (
                        environment.amount
                        + (
                                environment.price_ary[environment.day] * environment.stocks
                        ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets
###############################################################################################################################
def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    # read parameters
    cwd = kwargs.get('cwd', './' + str(model_name))

    if drl_lib == 'elegantrl':
        break_step = kwargs.get('break_step', 1e6)
        erl_params = kwargs.get('erl_params')

        agent = DRLAgent_erl(env=env,
                             price_array=price_array,
                             tech_array=tech_array,
                             turbulence_array=turbulence_array)

        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(model=model,
                                          cwd=cwd,
                                          total_timesteps=break_step)

    elif drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env=env,
                                     price_array=price_array,
                                     tech_array=tech_array,
                                     turbulence_array=turbulence_array)

        model, model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        # ray.shutdown()
        trained_model = agent_rllib.train_model(model=model,
                                                model_name=model_name,
                                                model_config=model_config,
                                                total_episodes=total_episodes)
        trained_model.save(cwd)


    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6)
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(model=model,
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(cwd)
        print('Trained model saved in ' + str(cwd))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')
############################################################################################################################

