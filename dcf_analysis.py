###DCF Analysis
#CREATED BY: SUNGJAE PARK
#SOURCE: https://valutico.com/discounted-cash-flow-analysis-your-complete-guide-with-examples/
#        https://corporatefinanceinstitute.com/resources/valuation/dcf-formula-guide/
#        https://mergersandinquisitions.com/dcf-model/

import pandas as pd
import yfinance as yf
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
import datetime
import time
import requests

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('future.no_silent_downcasting', True)

##Asking for stock ticker
def Ticker_Checker():
    try:
        stock_ticker = input("Please enter the stock ticker\n")
        stocks = yf.Ticker(stock_ticker)
        name = stocks.info['longName']
        return stock_ticker

    except AttributeError:
        print("ERROR: Invalid stock ticker")
        return Ticker_Checker()

    except requests.exceptions.HTTPError:
        print("ERROR: Invalid stock ticker")
        return Ticker_Checker()

stocks = Ticker_Checker()
stock_ticker = yf.Ticker(stocks)
company_name = stock_ticker.info['longName']


##Asking for benchmark index
benchmark_dict = {"s" : "SPY", "n" : "QQQ", "r" : "IWD"}
def Benchmark_Query():
    try:
        response = input("Please choose the benchmark index.\nType in : 's' for S&P 500 | 'n' for NASDAQ | 'r' for Russell\n").lower()
        benchmark_ticker = benchmark_dict[response]
        return benchmark_ticker

    except (NameError, KeyError) as e:
        print("ERROR: Invalid response")
        return Benchmark_Query()

benchmark_ticker = Benchmark_Query()
benchmark_yf_ticker = yf.Ticker(benchmark_ticker)
benchmark_name = benchmark_yf_ticker.info['longName']


##Get stock data from yfinance
df_stock = yf.download(stocks)
df_stock = pd.DataFrame(df_stock.stack(future_stack = True).reset_index().rename(
    index = str,
    columns = {"level_1" : "Ticker", "Date" : "date", "Ticker" : "ticker", "Adj Close" : "adj_close", "Close" : "close",
               "High" : "high", "Low" : "low", "Open" : "open",  "Volume" : "volume"}).sort_values(['ticker', 'date']))

benchmark = benchmark_ticker
df_benchmark = yf.download(benchmark)
df_benchmark = pd.DataFrame(df_benchmark.stack(future_stack = True).reset_index().rename(
    index = str,
    columns = {"level_1" : "Ticker", "Date" : "date", "Ticker" : "ticker", "Adj Close" : "adj_close", "Close" : "close",
               "High" : "high", "Low" : "low", "Open" : "open", "Volume" : "volume"}).sort_values(['ticker', 'date']))


##Filtering - date filtering to deal with n/a on the first row
stock_start = df_stock["date"].iloc[0].date()
benchmark_start = df_benchmark["date"].iloc[0].date()
stock_today = df_stock["date"].iloc[-1].date()

print(f"Our record of {company_name} begins from {stock_start}")
print(f"Our record of {benchmark_name} begins from {benchmark_start}")
time.sleep(1.5)


#Date parameter query
base_y = int(input("When should be the start date of the stock and benchmark data?\nPlease provide the [year] for the date parameter"))
base_m = int(input("Please provide the [month] for the date parameter"))
base_d = int(input("Please provide the [day] for the date parameter"))

base_full = datetime.datetime(base_y, base_m, base_d)
base_parameter = base_full.strftime("%Y-%m-%d")

print(f"The date parameter is '{base_parameter}'")
time.sleep(1.5)

df_stock["stock_daily_gain"] = df_stock["close"].pct_change()
df_benchmark["benchmark_daily_gain"] = df_benchmark["close"].pct_change()
df_stock = df_stock.loc[df_stock["date"] > base_parameter]
df_benchmark = df_benchmark.loc[df_benchmark["date"] > base_parameter]


##Asking for investment time frame
inv_time_frame = int(input(f"Please type in the number of years you will invest in {company_name} for"))
start_year = int(stock_today.year)-inv_time_frame

start_date = datetime.datetime(start_year, int(stock_today.month), int(stock_today.day))
beta_start_date = start_date.strftime("%Y-%m-%d")
df_stock_b = df_stock[df_stock["date"] > beta_start_date].reset_index()
df_benchmark_b = df_benchmark[df_benchmark["date"] > beta_start_date].reset_index()


##Asking for a beta calculation method: regression method vs. covariance method
beta = ""
beta_response = ""

#regression method
def Get_Beta(x, y):
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    except TypeError:
        return "ERROR: Invalid date parameter"

reg_beta = Get_Beta(df_benchmark_b["benchmark_daily_gain"], df_stock_b["stock_daily_gain"])

if type(reg_beta) == str:
    print(beta)
    time.sleep(1.5)
    exit()

#covariance method
cov_beta = df_stock_b.stock_daily_gain.cov(df_benchmark_b.benchmark_daily_gain) / np.var(df_benchmark_b.benchmark_daily_gain)

print(f"Here are the betas using different methods:\nRegression method: {reg_beta}\nCovariance method: {cov_beta}")

while beta_response not in ["r", "c"]:
    beta_response = str(input("Please choose the beta calculation method. Type in : 'r' for Regression method | 'c' for Covariance method"))

if beta_response == "r":
    beta = reg_beta
elif beta_response == "c":
    beta = cov_beta


##Calculate capital annual growth rate (CAGR)
rf_irx = yf.download("^IRX")
rf_irx = pd.DataFrame(rf_irx)
df_rf = rf_irx["Close"].iloc[::-1]
rf_rate = df_rf.iloc[0].iloc[0]
rf_rate = rf_rate / 100

num_years = int(df_stock["date"].iloc[-1].year) - int(df_stock["date"].iloc[0].year)
mkt_rate = (df_benchmark["close"].iloc[-1] / df_benchmark["close"].iloc[0])**(1 / num_years) - 1

cagr = (rf_rate + beta * (mkt_rate - rf_rate))


##Prepare financial statements
df_is = pd.DataFrame(stock_ticker.financials)
df_bs = pd.DataFrame(stock_ticker.balance_sheet)
df_cf = pd.DataFrame(stock_ticker.cashflow)

#drop columns beyond year 4
def Drop_Beyond_Year4(df, col_num):

    current_col_num = len(df.columns)

    if current_col_num <= col_num:
        return df

    drop_col_num = current_col_num - col_num
    df = df.iloc[:, :col_num]
    return df

df_is = Drop_Beyond_Year4(df_is, 4)
df_bs = Drop_Beyond_Year4(df_bs, 4)
df_cf = Drop_Beyond_Year4(df_cf, 4)

df_fs = pd.concat([df_is, df_cf, df_bs])
df_fs = df_fs.rename(columns = {df_fs.columns[0] : "y4", df_fs.columns[1] : "y3",
                                df_fs.columns[2] : "y2", df_fs.columns[3] : "y1",})
df_fs = df_fs.reset_index(names = "line")
df_fs["line"] = df_fs["line"].str.replace(" ", "_")
df_fs["line"] = df_fs["line"].str.strip()


##Change line item names using line item dictionary
line_names_response = ""
line_names = {"ebit" : ["EBIT", "Earnings_Before_Interest_And_Taxes"], "depr_amort" : ["Depreciation_And_Amortization"],
              "work_cap" : ["Change_In_Working_Capital"], "capex" : ["Capital_Expenditure", "CAPEX"],
              "equity" : ["Stockholders_Equity", "Stockholder's_Equity", "Total_Equity"], "asset" : ["Total_Assets"],
              "int_exp" : ["Interest_Expense"], "debt" : ["Total_Debt"], "shares" : ["Share_Issued"]}

df_line_names = pd.DataFrame.from_dict(line_names, orient = 'index')
print(df_line_names)

while line_names_response not in ["y", "n"]:
    line_names_response = str(input(f"The columns to the right should match with FSLIs from the financial statements of "
                                    f"{company_name}. Proceed with the suggested changes? - ['y' / 'n']"))

if line_names_response == 'n':
    print("Please revise the 'line_names' dictionary and re-run the code")
    time.sleep(1.5)
    exit()
elif line_names_response == 'y':
    print("\nProcessing...")

line_column = "line"

for index, row in df_fs.iterrows():
    for key, value in line_names.items():
        for i in value:
            if row["line"] == i:
                df_fs.replace({line_column : {i : key}}, inplace = True)


##Create a separate line item table for cash flow calculation
cf_dollar = df_fs[df_fs["line"].isin(["ebit", "depr_amort", "work_cap", "capex", "equity",
                                     "asset", "int_exp", "debt", "shares"])].set_index("line").fillna(0)

print(cf_dollar)
##Calculate Weighted Average Cost of Capital (WACC)
equity_c = cagr
equity_w = cf_dollar.loc["equity"].iloc[0] / cf_dollar.loc["asset"].iloc[0]
debt_c = cf_dollar.loc["int_exp"].iloc[0] / cf_dollar.loc["debt"].iloc[0]
debt_w = (cf_dollar.loc["asset"].iloc[0] - cf_dollar.loc["equity"].iloc[0]) / cf_dollar.loc["asset"].iloc[0]
tax_rate = 0.21

wacc = equity_c * equity_w + (debt_c * debt_w) * (1 - tax_rate)


##Calculate free cash flow

#growth rate
reversed_columns = list(cf_dollar.columns[::-1])
cf_dollar_reversed = cf_dollar[reversed_columns]

cf_growth = cf_dollar_reversed.pct_change(axis = 1)
cf_growth["growth_rate"] = cf_growth[cf_growth.columns].mean(axis = 1, skipna = True)

cf_merged = pd.merge(cf_dollar_reversed, cf_growth[["growth_rate"]], on ="line", how ="left")

#future value
fv_cf = 0
i = 0
fv_cf_dc = {}

while i < 6:
    cash_flow_future = (cf_dollar.loc["ebit"].iloc[0] * ((1 + cf_growth.loc["ebit"]["growth_rate"])**i) * (1 - tax_rate) +
                        cf_dollar.loc["depr_amort"].iloc[0] * ((1 + cf_growth.loc["depr_amort"]["growth_rate"])**i) +
                         cf_dollar.loc["work_cap"].iloc[0] * ((1 + cf_growth.loc["work_cap"]["growth_rate"])**i) +
                         cf_dollar.loc["capex"].iloc[0] * ((1 + cf_growth.loc["capex"]["growth_rate"])**i))
    fv_cf_dc[i] = cash_flow_future
    fv_cf += cash_flow_future
    i += 1

#terminal value (perpetual growth rate assumed to be 2.5%)
perp_rate = 0.025
exit_x = stock_ticker.info["forwardPE"]
ebitda = cf_dollar.loc["ebit"].iloc[0] + cf_dollar.loc["depr_amort"].iloc[0]

tv_cf_1 = (fv_cf_dc[5] * (1 + perp_rate)) / (wacc - perp_rate)
tv_cf_2 = ebitda * exit_x

#discount cash flow
dcf = 0
total_disc_cf = 0
monthly_factor = int(df_cf.columns[1].month)

for i in range(len(fv_cf_dc)):
    disc_cf = fv_cf_dc[i] * (1 / ((1 + wacc)**(monthly_factor / 12 + i)))
    total_disc_cf += disc_cf

dcf = total_disc_cf + (tv_cf_2 * (1 / ((1 + wacc)**(monthly_factor / 12 + len(fv_cf_dc)))))
projected_share_price = dcf / cf_dollar.loc["shares"].iloc[0]

print(f"\nCompany name: {company_name}\nBenchmark used: {benchmark_name}\nStart date: {base_parameter}\nBeta: {beta}\n"
      f"WACC: {wacc}\nCalculated share price: {projected_share_price}")
