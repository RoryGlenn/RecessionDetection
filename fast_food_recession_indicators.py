# The Recession just hit Starbucks. CEO Warns: “People have stopped coming”
# https://www.youtube.com/watch?v=IXjgY845bmA


from secedgar import filings
from datetime import date

daily_filings = filings(
    start_date=date(2024, 1, 1), user_agent="romglenn@gmail.com"
)
daily_urls = daily_filings.get_urls()
from pprint import pprint


pprint(daily_urls)

