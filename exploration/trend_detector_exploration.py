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
  SUBSTRING(s.start_month_year FROM 1 FOR 7) AS month,
  COUNT(DISTINCT s.nct_id) AS trial_count
FROM studies s
JOIN conditions c ON s.nct_id = c.nct_id
WHERE SUBSTRING(s.start_month_year FROM 1 FOR 7) IN (
  '{start_dt.strftime("%Y-%m")}', '{end_dt.strftime("%Y-%m")}'
)
GROUP BY c.name, SUBSTRING(s.start_month_year FROM 1 FOR 7)
ORDER BY condition, month;
"""







df = get_table(query)


#import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embeddings
embeddings_path = os.path.join("data", "condition_embeddings.pkl")
with open(embeddings_path, "rb") as f:
    embeddings_data = pd.read_pickle(f)

condition_embeddings = embeddings_data["condition_embeddings"]
conditions_df = embeddings_data["conditions_df"]

# 🔧 Normalize both sets of condition names
df['condition'] = df['condition'].str.strip().str.lower()
conditions_df['condition'] = conditions_df['condition'].str.strip().str.lower()

# 🔁 Group similar conditions within each month
grouped = []

for month, group in df.groupby("month"):
    conditions = group['condition'].unique().tolist()
    mask = conditions_df['condition'].isin(conditions)
    group_conditions_df = conditions_df[mask].reset_index(drop=True)

    if group_conditions_df.empty:
        print(f"⚠️ Skipping month {month}: No embeddings matched for conditions:")
        print("Missing:", conditions)
        grouped.append(group)
        continue

    group_embeddings = condition_embeddings[mask.values]

    sim_matrix = cosine_similarity(group_embeddings)
    threshold = 0.85
    map_in_month = {}
    used = set()

    for i, cond in enumerate(group_conditions_df['condition']):
        if cond in used:
            continue
        map_in_month[cond] = cond
        for j in range(i + 1, len(group_conditions_df)):
            other = group_conditions_df['condition'][j]
            if sim_matrix[i, j] > threshold:
                map_in_month[other] = cond
                used.add(other)

    group['condition'] = group['condition'].map(map_in_month)
    grouped.append(group)

# ✅ Combine groups and proceed with aggregation
df = pd.concat(grouped).reset_index(drop=True)
import numpy as np

# Group in case semantic mapping was done
df = df.groupby(['condition', 'month'], as_index=False)['trial_count'].sum()

# Pivot to wide format: rows = condition, columns = months
pivot = df.pivot(index="condition", columns="month", values="trial_count").fillna(0)

# Ensure only 2 months are present
if len(pivot.columns) != 2:
    raise ValueError(f"Expected 2 months, but found: {pivot.columns.tolist()}")

# Rename columns
pivot.columns = ["First_User_Month", "Second_User_Month"]

# Add delta and safe percent change
pivot["delta"] = pivot["Second_User_Month"] - pivot["First_User_Month"]
pivot["pct_change"] = np.where(
    pivot["First_User_Month"] == 0,
    np.nan,
    100 * pivot["delta"] / pivot["First_User_Month"]
)
pivot["is_new_condition"] = pivot["First_User_Month"] == 0

# Reset index for flat DataFrame
pivot = pivot.reset_index()

# Get top 100 increases and decreases (same logic as SummaryAgent)
increases = pivot[pivot["delta"] > 0].sort_values(
    by=["delta", "pct_change"], ascending=[False, False]
).head(100)

decreases = pivot[pivot["delta"] < 0].sort_values(
    by=["delta"], ascending=True
).head(100)


# Create payload from the trend logic
trend_payload = (
    increases.head(5).to_dict(orient="records") +
    decreases.head(5).to_dict(orient="records")
)


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
                    "description": "Bullet points highlighting specific condition-level trends. Use numbers and percentages to quantify changes."
                }
            },
            "required": ["summary", "highlights"]
        }
    }
]

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
        f"Each item contains the condition name, the number of trials in the first and second month, and the delta and percent change. "
        f"Please summarize this trend data into highlights. Say the first number and the last number so the user knows specifics..\n\n"
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