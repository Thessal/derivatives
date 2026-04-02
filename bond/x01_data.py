### Data processing code from x01_data.ipynb

import pandas as pd 
import numpy as np 

def parse_macroevent(filename):
    cols = ["date","country","event","currency","previous","estimate","actual","impact","source"]
    df = pd.read_csv(filename, names=cols, index_col=["event"])[["date", "actual"]].groupby("event").last()
    df.columns = ["Settle Date", "Bond Yield"]
    df["Settle Date"] = pd.to_datetime([x.date() for x in pd.to_datetime(df["Settle Date"])])
    df.index = df.index.str.extract(r'(\d+)', expand=False).astype(int).rename("Maturity")
    return df.sort_index()

def bond_pricing(notional, payment_per_year, coupon_rate, accepted_yield, issue_date, maturity_date, settlement_date):
    months = {1:12, 2:6, 4:3, 12:1}[payment_per_year]
    terms = pd.date_range(issue_date, maturity_date, freq=pd.DateOffset(months=months))
    df_cashflow = pd.DataFrame({
    "date": terms,
    "index": range(len(terms)),
    "cashflow": float(0.0),
    }).set_index("date")
    df_cashflow["discount"] = df_cashflow["index"].apply(lambda n : 1/(1+accepted_yield/payment_per_year)**n)
    df_cashflow.loc[maturity_date,"cashflow"] += notional
    df_cashflow.loc[terms[1:],"cashflow"] += notional * coupon_rate / payment_per_year
    issue_date_pv = df_cashflow.eval("cashflow * discount").sum()
    df_cashflow.loc[issue_date, "cashflow"] -= issue_date_pv

    coupon_date = terms[1]
    if settlement_date >= coupon_date:
        return None, np.nan
    if settlement_date == issue_date:
        dirty_price = issue_date_pv
    elif settlement_date > issue_date:
        a = (coupon_date - settlement_date).days
        b = (coupon_date - issue_date).days
        PV_at_coupon_date = issue_date_pv * (1+accepted_yield/payment_per_year)
        dirty_price = PV_at_coupon_date / (1+accepted_yield/payment_per_year * a/b)
    elif settlement_date < issue_date:
        a = (issue_date - settlement_date).days
        b = (issue_date - (issue_date - pd.DateOffset(months=months))).days
        dirty_price = issue_date_pv / (1+accepted_yield/payment_per_year * a/b) # Is it a standard way to do this? 
    else:
        raise Exception()

    return df_cashflow, dirty_price
