package main

import (
    "fmt"
    "text/template"
    "os"
)


type Function struct {
    Name       string `json:"name"`
    Description string `json:"description"`
    Parameters map[string]interface{} `json:"parameters"`
}

type Tool struct {
    Function Function `json:"function"`
}

type ToolCall struct {
    Function struct {
        Name       string `json:"name"`
        Arguments  string `json:"arguments"`
    } `json:"function"`
}

type Message struct {
    Role       string      `json:"role"`
    Content    string      `json:"content"`
    ToolCalls  []ToolCall  `json:"tool_calls"`
}

type Data struct {
    Messages  []Message   `json:"messages"`
    Tools     []Tool      `json:"tools"`
    System    string      `json:"system"`
    Prompt    string      `json:"prompt"`
    Response  string      `json:"response"`
}

func main() {
	const tpl = `
{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. If available tools are not relevant in assisting with user query, just respond in natural conversational language. Don't make assumptions about what values to plug into functions. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags.
<tools>
{{- range .Tools }}
{{ .Function.Name }}: {{ .Function.Description }}
Parameters: {{ .Function.Parameters }}
{{- end }}
</tools>
For each function call return a JSON object, with the following pydantic model json schema for each:
{'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
Each function call should be enclosed within <tool_call> </tool_call> XML tags. You must use <scratch_pad> </scratch_pad> XML tags to record your reasoning and planning before you call the functions as follows.
Example:
<scratch_pad>
Goal: <state task assigned by user>
Actions:
<if tool calls need to be generated:>
- {result_var_name1} = functions.{function_name1}({param1}={value1},...)
- {result_var_name2, result_var_name3} = ...
<if no tool call needs to be generated:> None
Observation: <set observation 'None' with tool calls; plan final tools results summary when provided>
Reflection: <evaluate query-tool relevance and required parameters when tools called; analyze overall task status when observations made>
</scratch_pad>
<tool_call>
{'name': <function-name>, 'arguments': <args-dict>}
</tool_call>
{{- end }}<|im_end|>
{{- end }}

{{- $hasToolResponses := false }}
{{- range .Messages }}
{{- if eq .Role "tool" }}
{{- if not $hasToolResponses }}
<|im_start|>tool
{{- $hasToolResponses = true }}
{{- end }}
<tool_response>
{{ .Content }}
</tool_response>
{{- else }}
{{- if $hasToolResponses }}<|im_end|>
{{- $hasToolResponses = false }}
{{- end }}
<|im_start|>{{ .Role }}
{{- if and (eq .Role "assistant") .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- else }}
{{ .Content }}
{{- end }}<|im_end|>
{{- end }}
{{- end }}
{{- if $hasToolResponses }}<|im_end|>
{{- end }}

{{- else }}
{{- if .System }}
<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}

{{- if .Prompt }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}

<|im_start|>assistant
{{ .Response }}<|im_end|>
{{- end }}
`

 // Create sample data with multiple tool calls
 data := Data{
	Messages: []Message{
		{
			Role:    "user",
			Content: "What's the weather like in New York and LA today?",
		},
		{
			Role:    "assistant",
			Content: "To provide you with accurate information about the weather in New York and Los Angeles today, I'll need to check the current weather data for both cities.",
			ToolCalls: []ToolCall{
				{
					Function: struct {
						Name       string `json:"name"`
						Arguments  string `json:"arguments"`
					}{
						Name: "get_weather",
						Arguments: `{"location": "New York", "date": "today"}`,
					},
				},
				{
					Function: struct {
						Name       string `json:"name"`
						Arguments  string `json:"arguments"`
					}{
						Name: "get_weather",
						Arguments: `{"location": "Los Angeles", "date": "today"}`,
					},
				},
			},
		},
		{
			Role:    "tool",
			Content: `{"temperature": 72, "condition": "Partly cloudy", "humidity": 65}`,
		},
		{
			Role:    "tool",
			Content: `{"temperature": 85, "condition": "Sunny", "humidity": 30}`,
		},
		{
			Role:    "assistant",
			Content: "Based on the current weather data, the weather in New York today is partly cloudy with a temperature of 72°F and humidity at 65%. In Los Angeles, it's sunny with a temperature of 85°F and humidity at 30%.",
		},
	},
	Tools: []Tool{
		{
			Function: Function{
				Name: "get_weather",
				Description: "Get the current weather for a specific location",
				Parameters: map[string]interface{}{
					"location": "string",
					"date": "string",
				},
			},
		},
	},
	System:   "You are a helpful AI assistant specialized in weather forecasting.",
	Prompt:   "Please provide the weather details.",
	Response: "Here's the weather information.",
}

// Create a new template and parse the letter into it.
tmpl, err := template.New("output").Parse(tpl)
if err != nil {
	fmt.Println("Error creating template:", err)
	return
}

// Execute the template and write the output to stdout.
err = tmpl.Execute(os.Stdout, data)
if err != nil {
	fmt.Println("Error executing template:", err)
	return
}
}
