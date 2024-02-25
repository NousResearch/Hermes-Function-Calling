# Hermes-Function-Calling

## Install required packages
`pip install -r requirements.txt`

## Run function call inference with query
`python functioncall.py --model_path NousResearch/Qwen1.5-0.5b-hermes-func-calling --chat_template chatml --query "I need the current stock price of Tesla (TSLA)"`

### Command line arguments:
--model_path: Path to the model folder.
--chat_template: Chat template for prompt formatting (default: "chatml").
--num_fewshot: Option to subset eval dataset (default: None).
--load_in_4bit: Option to load in 4bit with bitsandbytes (default: "False").
--query: Query to be used for function call inference (default: "I need the current stock price of Tesla (TSLA)").

## Inference exmple
```
<|im_start|>user
Fetch the stock fundamentals data for Tesla (TSLA)<|im_end|>
<|im_start|>tool
<tool_response>
{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}
</tool_response>
<|im_end|>
<|im_start|>assistant
Here is the stock fundamentals data for Tesla (TSLA), as provided by the function:

```json
{
  "company_name": "Tesla, Inc.",
  "sector": "Consumer Cyclical",
  "industry": "Auto Manufacturers",
  "market_cap": 611384164352,
  "pe_ratio": 49.604652,
  "pb_ratio": 9.762013,
  "dividend_yield": None,
  "eps": 4.3,
  "beta": 2.427,
  "52_week_high": 299.29,
  "52_week_low": 152.37
}
```json
```