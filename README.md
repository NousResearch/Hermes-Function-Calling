# Hermes-Function-Calling

## Install requied packages
pip install -r requirements.txt

## Run function call inference with query
`python functioncall.py --model_path NousResearch/Qwen1.5-0.5b-hermes-func-calling --chat_template chatml --query "I need the current stock price of Tesla (TSLA)"`

```
<|im_start|>user
Fetch the fundamentals data for Tesla (TSLA) stock<|im_end|>
<|im_start|>assistant
<tool_call>
{'arguments': {'symbol': 'TSLA'}, 'name': 'get_current_stock_price'}
</tool_call><|im_end|>
2024-02-24:15:46:10,900 INFO     [functioncall.py:65] all validations passed
2024-02-24:15:46:10,900 INFO     [functioncall.py:66] parsed tool calls:
[
  {
    "arguments": {
      "symbol": "TSLA"
    },
    "name": "get_current_stock_price"
  }
]
191.97
```