import pandas as pd
import yfinance as yf
from typing import List
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

@tool
def get_current_stock_price(symbol: str) -> float:
  """
  Get the current stock price for a given symbol.

  Args:
    symbol (str): The stock symbol.

  Returns:
    float: The current stock price, or None if an error occurs.
  """
  try:
    stock = yf.Ticker(symbol)
    # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
    current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
    return current_price if current_price else None
  except Exception as e:
    print(f"Error fetching current price for {symbol}: {e}")
    return None

@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol using yfinance API.

    Args:
        symbol (str): The stock symbol.

    Returns:
        dict: A dictionary containing fundamental data.
            Keys:
                - 'symbol': The stock symbol.
                - 'company_name': The long name of the company.
                - 'sector': The sector to which the company belongs.
                - 'industry': The industry to which the company belongs.
                - 'market_cap': The market capitalization of the company.
                - 'pe_ratio': The forward price-to-earnings ratio.
                - 'pb_ratio': The price-to-book ratio.
                - 'dividend_yield': The dividend yield.
                - 'eps': The trailing earnings per share.
                - 'beta': The beta value of the stock.
                - '52_week_high': The 52-week high price of the stock.
                - '52_week_low': The 52-week low price of the stock.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals = {
            'symbol': symbol,
            'company_name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('forwardPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'eps': info.get('trailingEps', None),
            'beta': info.get('beta', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None)
        }
        return fundamentals
    except Exception as e:
        print(f"Error getting fundamentals for {symbol}: {e}")
        return {}

@tool
def get_financial_statements(symbol: str) -> dict:
    """
    Get financial statements for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing financial statements (income statement, balance sheet, cash flow statement).
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        return financials
    except Exception as e:
        print(f"Error fetching financial statements for {symbol}: {e}")
        return {}

@tool
def get_key_financial_ratios(symbol: str) -> dict:
    """
    Get key financial ratios for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing key financial ratios.
    """
    try:
        stock = yf.Ticker(symbol)
        key_ratios = stock.info
        return key_ratios
    except Exception as e:
        print(f"Error fetching key financial ratios for {symbol}: {e}")
        return {}

@tool
def get_analyst_recommendations(symbol: str) -> pd.DataFrame:
    """
    Get analyst recommendations for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing analyst recommendations.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations
    except Exception as e:
        print(f"Error fetching analyst recommendations for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_dividend_data(symbol: str) -> pd.DataFrame:
    """
    Get dividend data for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing dividend data.
    """
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        return dividends
    except Exception as e:
        print(f"Error fetching dividend data for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_company_news(symbol: str) -> pd.DataFrame:
    """
    Get company news and press releases for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing company news and press releases.
    """
    try:
        news = yf.Ticker(symbol).news
        return news
    except Exception as e:
        print(f"Error fetching company news for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_technical_indicators(symbol: str) -> pd.DataFrame:
    """
    Get technical indicators for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing technical indicators.
    """
    try:
        indicators = yf.Ticker(symbol).history(period="max")
        return indicators
    except Exception as e:
        print(f"Error fetching technical indicators for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_company_profile(symbol: str) -> dict:
    """
    Get company profile and overview for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing company profile and overview.
    """
    try:
        profile = yf.Ticker(symbol).info
        return profile
    except Exception as e:
        print(f"Error fetching company profile for {symbol}: {e}")
        return {}
    
@tool
def execute_code(code_string):
    """
    Execute the provided Python code string using exec.

    Parameters:
    - code_string (str): The Python code to be executed.

    Returns:
    None

    Note:
    Use this function with caution, as executing arbitrary code can pose security risks.
    """
    try:
        exec(code_string)
    except Exception as e:
        print(f"An error occurred: {e}")

def get_openai_tools() -> List[dict]:
    functions = [
        get_current_stock_price,
        get_company_news,
        get_company_profile,
        get_stock_fundamentals,
        get_financial_statements,
        get_key_financial_ratios,
        get_analyst_recommendations,
        get_dividend_data,
        #get_historical_price_data,
        get_technical_indicators,
        execute_code
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools