from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 3 + 2
Output: 3 + 2 is 5.

Input: 3 * 10
Output: 3 * 10 is 30. Fun fact you can even multiply 10 * 3 which gives same result.

Input: Why is the color of sky?
Output: Bruh? You alright? Is it maths query?
"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        { "role": "system", "content": system_prompt },
        # { "role": "user", "content": "what is 9 * 8" }, # 9 * 8 is 72. This means if you have 9 groups of 8 items each, you would have a total of 72 items.
        { "role": "user", "content": "what is a computer?" }, # I'm sorry, but I can only provide help with maths-related queries. Please ask a maths question.
    ]
)

print(result.choices[0].message.content)