from typing import List
import pandas as pd
import yfinance as yf
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_tool

@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol using yfinance API.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: A dictionary containing fundamental data.
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
def get_historical_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get historical price data for a given stock symbol.

    Args:
    symbol (str): The stock symbol.
    start_date (str): The start date for historical data (YYYY-MM-DD).
    end_date (str): The end date for historical data (YYYY-MM-DD).

    Returns:
    pd.DataFrame: DataFrame containing historical price data.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching historical price data for {symbol}: {e}")
        return pd.DataFrame()

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
def get_option_chain(symbol: str) -> pd.DataFrame:
    """
    Get option chain data for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing option chain data.
    """
    try:
        option_chain = yf.Ticker(symbol).option_chain
        return option_chain
    except Exception as e:
        print(f"Error fetching option chain data for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_market_indices() -> pd.DataFrame:
    """
    Get data for major market indices.

    Returns:
    pd.DataFrame: DataFrame containing data for major market indices.
    """
    try:
        indices = yf.download('^GSPC ^DJI ^IXIC', start="2000-01-01", end="2023-01-01")
        return indices
    except Exception as e:
        print(f"Error fetching market indices data: {e}")
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

def get_openai_tools() -> List[dict]:
    functions = [
        get_company_news,
        get_company_profile,
        get_stock_fundamentals,
        get_financial_statements,
        get_key_financial_ratios,
        get_analyst_recommendations,
        get_dividend_data,
        get_historical_price_data,
        get_market_indices,
        get_technical_indicators,
        get_option_chain
    ]

    tools = [format_tool_to_openai_tool(f) for f in functions]
    return tools