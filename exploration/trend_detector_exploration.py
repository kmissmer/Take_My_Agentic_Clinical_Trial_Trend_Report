"""
trend_detector_explainer.py

This script is a whiteboard for the TrendDetectorAgent — it walks through the process of pulling condition trend data 
from the AACT database, computing changes, and generating a human-readable explanation using OpenAI function calling. 
Used for testing and development.
"""



"""
WIP
"""


import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv("base_dir")
os.chdir(base_dir)
sys.path.append(base_dir)

from utils.sql_util import get_table
from utils.openai_util import get_azure_openai_client
from datetime import datetime, timedelta


azure_deployment = "gpt-4o-mini"

# Example input
start_month = 1
start_year = 2020
end_month = 4
end_year = 2025

# Convert to date strings
start_date = f"{start_year}-{start_month}-01"
start_dt = datetime.strptime(start_date, "%Y-%m-%d")
end_date = f"{end_year}-{end_month}-01"
end_dt = datetime.strptime(end_date, "%Y-%m-%d")


# Dates as strings for SQL
start_month_year = start_dt.strftime("%Y-%m-%d")
end_month_year = end_dt.strftime("%Y-%m-%d")

# Also get nice strings for GPT
start_label = start_dt.strftime("%B %Y")  # e.g. March 2025
end_label = end_dt.strftime("%B %Y")  # e.g. April 2025

# 1. Fetch and prepare trend data
query = f"""
SELECT
  c.name AS condition,
  s.start_month_year AS month,
  COUNT(DISTINCT s.nct_id) AS trial_count
FROM studies s
JOIN conditions c ON s.nct_id = c.nct_id
WHERE s.start_month_year IN ('{start_month_year}', '{end_month_year}')
GROUP BY c.name, s.start_month_year
ORDER BY condition, month;
"""


df = get_table(query)

pivot = df.pivot(index="condition", columns="month", values="trial_count").fillna(0)
pivot.columns = ["First_User_Month", "Second_User_Month"]
pivot["delta"] = pivot["Second_User_Month"] - pivot["First_User_Month"]
pivot["pct_change"] = 100 * pivot["delta"] / (pivot["First_User_Month"] + 1)

top = pivot.sort_values("delta", ascending=False).head(100).reset_index()

# 2. Create the payload for GPT
trend_payload = top.to_dict(orient="records")

# 3. Define function calling schema
functions = [
    {
        "name": "explain_condition_trends",
        "description": "Generate a narrative explanation of the condition trends in clinical trials.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A plain-language summary of the most notable increases or decreases in trial activity."
                },
                "highlights": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Bullet points highlighting specific condition-level trends."
                }
            },
            "required": ["summary", "highlights"]
        }
    }
]

print("Key:", os.getenv("azure_openai_key"))
print("Endpoint:", os.getenv("azure_openai_endpoint"))

# 4. Use OpenAI to summarize
client = get_azure_openai_client()

print("Client type:", type(client))


system_message = {
    "role": "system",
    "content": "You are a medical data analyst who summarizes clinical trial trends in plain English."
}
user_message = {
    "role": "user",
    "content": (
        f"Here are clinical trial activity changes by condition between {start_label} and {end_label}. "
        f"Please summarize the trend data below:\n\n"
        f"{json.dumps(trend_payload, indent=2)}"
    )
}


response = client.chat.completions.create(
    model=azure_deployment,
    messages=[system_message, user_message],
    functions=functions,
    function_call={"name": "explain_condition_trends"},
    temperature=0.7,
    max_tokens=2048
)

# Safely extract and print the structured response
msg = response.choices[0].message

if msg.function_call and msg.function_call.arguments:
    output = json.loads(msg.function_call.arguments)
    print("Summary:\n")
    print(output["summary"])
    print("\nHighlights:")
    for item in output["highlights"]:
        print("-", item)
else:
    print("❌ GPT did not return a valid function call.")
    print("Message content:", msg.content or "[No content]")



# Example output
"""
Between January 2020 and April 2025, there has been a significant increase in clinical trial activity across various medical conditions. Some conditions have seen particularly notable rises in trial participation.

Highlights:
- Stroke trials increased from 9 to 15, a 60% rise.
- Atrial Fibrillation saw a jump from 0 to 4 trials, representing a 400% increase.
- Child Development, Cesarean Section Complications, Obesity-related conditions, and several others all started with no trials and reached 3 trials, indicating a 300% increase each.
- Stress trials increased from 1 to 4, a 150% rise.
- Healthy condition trials grew from 6 to 9, up 42.9%.
- Many conditions with previously no trials, such as Readmission and Degenerative Disc Disease, have now recorded 2 trials each, showing a 200% rise.
"""