# The Recession just hit Starbucks. CEO Warns: “People have stopped coming”
# https://www.youtube.com/watch?v=IXjgY845bmA
from pprint import pprint
import pandas as pd

import os
import re
from requests.exceptions import RequestException
from datetime import date
from secedgar import CompanyFilings, FilingType


def fetch_filings(ticker, company_name, email, start_date, end_date):
    save_path = f"./data/{company_name}"
    try:
        filings = CompanyFilings(
            cik_lookup=ticker,
            filing_type=FilingType.FILING_10K,
            user_agent=email,
            start_date=start_date,
            end_date=end_date,
            count=1,
        )
        filings.save(save_path)
    except RequestException as e:
        print(f"Request Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_filing_content(path):
    """Reads the content of the first text file found in the specified directory."""
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filename = path + "/" + filename
            with open(filename, "r", encoding="utf-8") as file:
                return file.read()


def extract_revenue_data(text):
    """Extracts revenue data from the filing text using a regular expression."""
    revenue_pattern = re.compile(r"Total revenue \$([\d,]+)", re.IGNORECASE)
    results = revenue_pattern.findall(text)
    if results:
        return float(results[0].replace(",", ""))


def calculate_yoy_growth(current_year_revenue, previous_year_revenue):
    """Calculates the Year-over-Year growth given two revenue figures."""
    if previous_year_revenue == 0:
        return None
    return (
        (current_year_revenue - previous_year_revenue) / previous_year_revenue
    ) * 100


def extract_tables_from_html(html_file_path):
    """
    Extracts tables from an HTML file and saves them as CSV files.

    Parameters:
        html_file_path (str): The path to the HTML file.

    Returns:
        A list of DataFrames, each representing a table extracted from the HTML file.
    """
    # Read the HTML file
    tables = pd.read_html(html_file_path)

    # Loop through the tables and optionally perform cleaning or further processing
    for index, df in enumerate(tables):
        # Print the table to check its content
        print(f"Table {index}:")
        print(df.head())

        # Save each table to a separate CSV file for further analysis if needed
        df.to_csv(f"table_{index}.csv", index=False)

    return tables


def clean_table(df):
    # Assuming the first row might be usable as a header if no header is present
    if (
        df.columns[0] != "Category"
    ):  # Checking if the headers are already set, which you might customize
        df.columns = [
            "Category",
            "2023",
            "2022",
            "2021",
        ]  # Adjust the number of columns as per the actual data
        df = df.iloc[
            1:
        ]  # Skip the original header row which is now set as column names
    df.reset_index(drop=True, inplace=True)
    return df


def find_sales_growth_table(tables):
    target_keywords = {
        "U.S. Company-owned stores",
        "U.S. franchise stores",
        "U.S. stores",
        "International stores",
    }

    for df in tables:
        # Check if any of the target keywords are in any of the DataFrame's rows
        if any(
            df.apply(
                lambda row: row.astype(str)
                .str.contains("|".join(target_keywords))
                .any(),
                axis=1,
            )
        ):
            # Optional: Clean and reformat this table as needed
            return df


# Cleaning and restructuring the DataFrame
def clean_table(df):
    # Selecting only the columns with actual data (assuming these are at indexes 2, 4, 6)
    df = df[[0, 2, 4, 6]]
    # Drop the first row that contains only None or column years (after setting proper headers)
    df.columns = ["Category", "2023", "2022", "2021"]
    df = df.drop(index=0)
    df = df.drop(index=1)  # Drop the year row which is now set as column names
    # Reset the index for cleanliness
    df.reset_index(drop=True, inplace=True)
    # Convert percentage strings to numerical values
    df.replace({"%": "", "\\+": "", "\\(": "-", "\\)": ""}, regex=True, inplace=True)
    return df


def clean_unknown_structure_df(df: pd.DataFrame):
    # Identify non-empty columns and drop completely empty ones
    df = df.dropna(how="all")

    # Drop columns where all elements are NA
    df = df.dropna(axis=1, how="all")

    # Function to check if a value is numeric
    def is_numeric(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    # Apply the is_numeric function to all rows except the first one and columns except the first one
    numeric_mask = df.iloc[1:, 1:].applymap(is_numeric).any()

    # Add the first column manually to the mask to retain it
    numeric_mask = pd.Series([True], index=[df.columns[0]]).append(numeric_mask)

    # Use the mask to filter columns
    df_filtered = df.loc[:, numeric_mask]

    # Attempt to find a header row if not set
    if (
        df.columns.dtype == "int"
    ):  # Check if columns are just numerical indices, suggesting headers might be in a row
        for i, row in df.iterrows():
            # Heuristic: If a row is all strings and the next row has fewer strings, it might be a header
            if row.apply(lambda x: isinstance(x, str)).all() and (
                i + 1 < len(df)
                and df.iloc[i + 1].apply(lambda x: isinstance(x, str)).sum() < row.size
            ):
                df.columns = row  # Set this row as the header
                df = df.drop(index=i)
                break

    # Standardize data entries
    df.replace({"%": "", "\\+": "", "\\(": "-", "\\)": ""}, regex=True, inplace=True)

    # Convert data columns to numeric if possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass  # Skip columns that can't be converted to numeric

    # Reset index after potential row deletions
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    filing_dict = {
        "SBUX": "Starbucks",
        "MCD": "McDonalds",
        "WEN": "Wendy's",
        "CMG": "Chipotle",
        # "YUM": "Yum Brands",
        # "QSR": "Restaurant Brands",
        # "DPZ": "Domino's",
        # "PZZA": "Papa John's",
        # "JACK": "Jack in the Box",
        # "QSR": "Restaurant Brands",
    }

    # get the filings for the companies
    for ticker, rest_name in filing_dict.items():
        if not os.path.exists(f"./data/{rest_name}/{ticker}/10-K"):
            fetch_filings(
                ticker,
                rest_name,
                email="glennrory@gmail.com",
                start_date=date(2023, 1, 1),
                end_date=date.today(),
            )

    # find a specific table like 'Same store growth' or 'Revenue'
    filename = "data/Domino's/DPZ/10-K/0000950170-24-019725.html"
    tbls = pd.read_html(filename)
    df = tbls[8]

    filename = "data/Jack in the Box/JACK/10-K/0000807882-23-000020.html"
    tbls = pd.read_html(filename)
    df2 = tbls[8]

    df = find_sales_growth_table(tbls)
    clean_df = clean_unknown_structure_df(df)

    for ticker, company_name in filing_dict.items():
        filing_content = read_filing_content(f"./data/{company_name}/{ticker}/10-K")
        revenue_data = extract_revenue_data(filing_content)
        print(f"{company_name} Revenue: ${revenue_data} million")


if __name__ == "__main__":
    main()
