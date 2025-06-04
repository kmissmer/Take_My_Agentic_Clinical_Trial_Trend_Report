import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utils.sql_util import get_table
from utils.openai_util import get_azure_openai_client

class SummaryAgent:
    def __init__(self, start_year, start_month, end_year, end_month, azure_deployment="gpt-4o-mini"):
        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.azure_deployment = azure_deployment

        self.start_date = f"{self.start_year}-{self.start_month}-01"
        self.end_date = f"{self.end_year}-{self.end_month}-01"

        self.start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        self.start_label = self.start_dt.strftime("%B %Y")
        self.end_label = self.end_dt.strftime("%B %Y")

    def fetch_and_process(self):
        query = f"""
        SELECT
          c.name AS condition,
          SUBSTRING(s.start_month_year FROM 1 FOR 7) AS month,
          COUNT(DISTINCT s.nct_id) AS trial_count
        FROM studies s
        JOIN conditions c ON s.nct_id = c.nct_id
        WHERE SUBSTRING(s.start_month_year FROM 1 FOR 7) IN (
          '{self.start_dt.strftime("%Y-%m")}', '{self.end_dt.strftime("%Y-%m")}'
        )
        GROUP BY c.name, SUBSTRING(s.start_month_year FROM 1 FOR 7)
        ORDER BY condition, month;
        """

        df = get_table(query)

        # Load precomputed embeddings
        with open(os.path.join("data", "condition_embeddings.pkl"), "rb") as f:
            embeddings_data = pd.read_pickle(f)

        condition_embeddings = embeddings_data["condition_embeddings"]
        conditions_df = embeddings_data["conditions_df"]

        # Normalize
        df['condition'] = df['condition'].str.strip().str.lower()
        conditions_df['condition'] = conditions_df['condition'].str.strip().str.lower()

        grouped = []
        for month, group in df.groupby("month"):
            conditions = group['condition'].unique().tolist()
            mask = conditions_df['condition'].isin(conditions)
            group_conditions_df = conditions_df[mask].reset_index(drop=True)

            if group_conditions_df.empty:
                grouped.append(group)
                continue

            group_embeddings = condition_embeddings[mask.values]
            sim_matrix = cosine_similarity(group_embeddings)
            map_in_month = {}
            used = set()

            for i, cond in enumerate(group_conditions_df['condition']):
                if cond in used:
                    continue
                map_in_month[cond] = cond
                for j in range(i + 1, len(group_conditions_df)):
                    other = group_conditions_df['condition'][j]
                    if sim_matrix[i, j] > 0.85:
                        map_in_month[other] = cond
                        used.add(other)

            group['condition'] = group['condition'].map(map_in_month)
            grouped.append(group)

        df = pd.concat(grouped).reset_index(drop=True)
        df = df.groupby(['condition', 'month'], as_index=False)['trial_count'].sum()

        pivot = df.pivot(index="condition", columns="month", values="trial_count").fillna(0)
        if len(pivot.columns) != 2:
            raise ValueError(f"Expected 2 months, but got: {pivot.columns.tolist()}")

        pivot.columns = ["First_User_Month", "Second_User_Month"]
        pivot["delta"] = pivot["Second_User_Month"] - pivot["First_User_Month"]
        pivot["pct_change"] = np.where(
            pivot["First_User_Month"] == 0,
            np.nan,
            100 * pivot["delta"] / pivot["First_User_Month"]
        )
        pivot = pivot.reset_index()

        increases = pivot[pivot["delta"] > 0].sort_values(by=["delta", "pct_change"], ascending=[False, False]).head(100)
        decreases = pivot[pivot["delta"] < 0].sort_values(by=["delta"], ascending=True).head(100)

        return increases, decreases

    def summarize(self, increases, decreases):
        trend_payload = (
            increases.head(5).to_dict(orient="records") +
            decreases.head(5).to_dict(orient="records")
        )

        client = get_azure_openai_client()

        system_message = {
            "role": "system",
            "content": "You are a medical data analyst who summarizes clinical trial trends in plain English."
        }
        user_message = {
            "role": "user",
            "content": (
                f"Here are clinical trial activity changes by condition between {self.start_label} and {self.end_label}.\n"
                f"Each item includes the condition name, trial counts in both months, delta, and percent change.\n\n"
                f"{json.dumps(trend_payload, indent=2)}"
            )
        }

        functions = [
            {
                "name": "explain_condition_trends_increase_decrease",
                "description": "Generate a narrative explanation of condition trends, and give 5 key increases and decreases. Give the specific numbers for beginning and end for each increase and decrease and percentages when interesting. If percentage is infinate, dont talk about percentage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "increases": {"type": "array", "items": {"type": "string"}},
                        "decreases": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["summary", "increases", "decreases"]
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.azure_deployment,
            messages=[system_message, user_message],
            functions=functions,
            function_call={"name": "explain_condition_trends_increase_decrease"},
            temperature=0.7,
            max_tokens=2048
        )

        msg = response.choices[0].message
        if msg.function_call and msg.function_call.arguments:
            output = json.loads(msg.function_call.arguments)
            return output["summary"], output["increases"], output["decreases"]
        else:
            return "❌ GPT did not return a valid function call.", [], []

    def execute(self):
        increases, decreases = self.fetch_and_process()
        return self.summarize(increases, decreases)
