# GenAI Cohort Project

This repository contains code and resources for the GenAI (Generative AI) Cohort project.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/OmGameHub/GenAICohort.git
   cd GenAICohort
   ```

2. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   source venv/Scripts/activate

   # On Unix/MacOS
   python -m venv venv
   source venv/bin/activate

   pip install -r requirements.txt
   ```

### Topics Covered

#### 1. **Tokenization**: Understanding how text is converted into tokens for processing by OpenAI models.
- Displaying vocabulary size for specific models
- Encoding text into tokens using OpenAI's tokenizer
- Decoding tokens back into text


#### 2. **Vector Embedding**: Exploring how text is represented as vectors.
- Using OpenAI's embedding model to convert text into vectors

#### 3. **Chain of Thought Prompting**: Break down complex tasks into smaller steps and solve them sequentially.
```
python chat_chain_of_thought_prompt.py

## Example output
> 13 + 2 * 7
🤔: The user has presented a mathematical expression involving both addition and multiplication.
🤔: According to the order of operations, multiplication takes precedence over addition. Therefore, I should first evaluate 2 * 7 before proceeding to addition.
🤔: Start by calculating the multiplication: 2 * 7 = 14.
🤖: Now, I need to add 13 to the result of the multiplication: 13 + 14.
```

#### 4. **AI Agent with Tools**: An implementation of a function-calling agent that can use different tools.
- Uses start, plan, action, observe mode
- Supports tools like:
  - `get_weather`: Retrieves weather information for a given city
  - `run_command`: Executes system commands
  - `add`: Performs addition of two numbers

To run the AI agent:
```bash
python ai_agent.py
```

Example usage:
```
> What's the weather in London?
🧠: The user wants to know the weather in London.
🧠: I should use the get_weather tool to retrieve this information.
🔨 Tool Called: get_weather London
🤖: The weather in London is Partly cloudy +15°C.
```