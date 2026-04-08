import requests
import pandas as pd
from lxml import html
import io
import binascii
import re
import time 

with open("x01_cookie.key", "rt") as f:
    cookie = f.read()

def convert_encoded_string(hex_string):
    # Escape function call 
    hex_string = re.search(r"d\d\(\s*'(.*?)'\s*\)", hex_string).group(1)

    # Split by '%' and filter out empty strings
    segments = [s for s in hex_string.split('%') if s]
    
    # Extract the last 2 characters of each segment (the hex part)
    hex_values = "".join(s[-2:] for s in segments)
    
    # Convert hex to bytes, then decode to utf-8
    return binascii.unhexlify(hex_values).decode('utf-8')

def get_irs_data(datestr="20260407"):
    url = 'http://www.smbs.biz/Eng/Exchange/IRS.jsp'
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'http://www.smbs.biz',
        'Connection': 'keep-alive',
        'Referer': 'http://www.smbs.biz/Eng/Exchange/IRS.jsp',
        'Cookie': cookie,
        'Upgrade-Insecure-Requests': '1',
        'Priority': 'u=0, i'
    }
    data = f'StrSch_sYear=2025&StrSch_sMonth=10&StrSch_sDay=08&StrSch_eYear=2026&StrSch_eMonth=04&StrSch_eDay=07&StrSch_Year={datestr[:4]}&StrSch_Month={datestr[5:7]}&StrSch_Day={datestr[8:10]}&StrSchFull=2025.10.08&StrSchFull2=&2026.04.07StrSchFull3={datestr}'

    time.sleep(1) # being nice
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    # Parse HTML
    tree = html.fromstring(response.content)
    
    # Extract element by XPath
    # /html/body/div/div[4]/div[2]/form/div[9]/table
    elements = tree.xpath('/html/body/div/div[4]/div[2]/form/div[9]/table')
    
    if not elements:
        raise ValueError("Cannot find the table at the specified XPath.")
        
    table_html = html.tostring(elements[0], encoding='unicode')
    
    # Parse table into Pandas DataFrame
    df_list = pd.read_html(io.StringIO(table_html))
    df = df_list[0] if df_list else pd.DataFrame()

    # Decoding
    df = df.map(convert_encoded_string)
    df.columns = df.columns.map(convert_encoded_string)
    df = df.set_index("Tenure")
    return df 