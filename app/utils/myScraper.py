import pandas as pd 
import chardet   
from selenium import webdriver 
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from io import StringIO
import datetime 
import os
import pytz

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#determining the date
utc_now = datetime.datetime.now(datetime.timezone.utc)
today = utc_now.astimezone(pytz.timezone("Asia/Colombo"))

today_stem = today.strftime('%Y%m%d')
today = today.strftime('%Y-%m-%d')

#setting a headless browser for web scraping
chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Firefox(options=chrome_options)
driver.implicitly_wait(10)

urlo = 'https://www.cse.lk/pages/trade-summary/trade-summary.component.html'

#%%
try:
    driver.get(urlo)
    # print(driver.page_source)
    
    # selecting the data table from the web page
    # final_select = Select(driver.find_element_by_name('DataTables_Table_0_length'))
    final_select = Select(driver.find_element("name", 'DataTables_Table_0_length'))
    final_select.select_by_visible_text('All')
except Exception as e:
    print("An error occurred. Please check if the URL is valid and the element exists on the page.")
    print("Detailed error:", e)
    driver.quit()  # Close the driver in case of an error
    exit(1)  # Exit the script

WebDriverWait(driver, 3)

df = pd.read_html(StringIO(driver.page_source))[0]

df['Date'] = today

driver.close()

# Create the directory if it does not exist
os.makedirs(f'daily_dumps', exist_ok=True)

with open(f'daily_dumps/{today_stem}.csv', 'w') as f:
    df.to_csv(f, index=False, header=True) 
    
# Check if the file exists, if not, create an empty DataFrame and write it to the file
if not os.path.exists('cse.csv'):
    pd.DataFrame(columns=df.columns).to_csv('cse.csv')

# print(df.shape)

rawdata = open('cse.csv', 'rb').read()
result = chardet.detect(rawdata)
encoding = result['encoding']

old = pd.read_csv('cse.csv', encoding=encoding)

# print(old.shape)

# Concatenate the old and new data
combined = pd.concat([old, df])

# Sort by 'Date' in descending order
combined = combined.sort_values('Date', ascending=False)

#Drop duplicates
#new = new.drop_duplicates(subset=['Symbol', 'Date'])

# Drop duplicates based on 'Symbol' and 'Date', keeping the most recent data
combined = combined.drop_duplicates(subset=['Symbol', 'Date'], keep='first')

# Sort by 'Date' in descending order and then by 'Company Name' in ascending order
combined = combined.sort_values(['Date', 'Company Name'], ascending=[False, True])

print(combined)

#combined.empty

print("Current working directory:", os.getcwd())
print("Combined DataFrame is empty:", combined.empty)

try:
    combined.to_csv('cse.csv', index=False) # Save the updated data
except Exception as e:
    print("Exception occurred:", e)

# with open("cse.json", "w") as f:
#     new.to_json(f)
