import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from utils.sql_util import get_table
from utils.openai_util import get_azure_openai_client
import numpy as np


class SummaryAgent:
    def __init__(self, start_year, start_month, end_year, end_month, azure_deployment="gpt-4o-mini"):
        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.azure_deployment = azure_deployment

        # Prepare start and end date strings
        self.start_date = f"{self.start_year}-{self.start_month}-01"
        self.end_date = f"{self.end_year}-{self.end_month}-01"

        self.start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        # Format for SQL queries and GPT
        self.start_month_year = self.start_dt.strftime("%Y-%m-%d")
        self.end_month_year = self.end_dt.strftime("%Y-%m-%d")
        self.start_label = self.start_dt.strftime("%B %Y")  # e.g. March 2025
        self.end_label = self.end_dt.strftime("%B %Y")  # e.g. April 2025

    def fetch_trend_data(self):
        query = f"""
        SELECT
        c.name AS condition,
        s.start_month_year AS month,
        COUNT(DISTINCT s.nct_id) AS trial_count
        FROM studies s
        JOIN conditions c ON s.nct_id = c.nct_id
        WHERE s.start_month_year IN ('{self.start_month_year}', '{self.end_month_year}')
        GROUP BY c.name, s.start_month_year
        ORDER BY condition, month;
        """
        # Fetch the data
        df = get_table(query)

        # Pivot to wide format
        pivot = df.pivot(index="condition", columns="month", values="trial_count").fillna(0)
        pivot.columns = ["First_User_Month", "Second_User_Month"]

        # Add delta
        pivot["delta"] = pivot["Second_User_Month"] - pivot["First_User_Month"]

        # Add pct_change — safe way (avoids divide by zero)
        pivot["pct_change"] = np.where(
            pivot["First_User_Month"] == 0,
            np.nan,  # or float('inf') if you want to flag new conditions
            100 * pivot["delta"] / pivot["First_User_Month"]
        )

        # Optional: flag new conditions
        pivot["is_new_condition"] = pivot["First_User_Month"] == 0

        # Reset index so condition is a column again
        pivot = pivot.reset_index()

        # Top 100 increases and decreases
        increases = pivot[pivot["delta"] > 0].sort_values(
            by=["delta", "pct_change"], ascending=[False, False]
        ).head(100)

        decreases = pivot[pivot["delta"] < 0].sort_values(
            by=["delta"], ascending=True
        ).head(100)

        return increases, decreases


    def generate_increases(self, top):
        increases = []
        for _, row in top.iterrows():
            increase = f"Condition: {row['condition']} | Change in trials: {row['delta']} trials ({row['pct_change']:.2f}% change)"
            increases.append(increase)
        return increases
    
    def generate_decreases(self, top):
        decreases = []
        for _, row in top[top['delta'] < 0].iterrows():
            decrease = f"Condition: {row['condition']} | Change in trials: {row['delta']} trials ({row['pct_change']:.2f}% change)"
            decreases.append(decrease)
        return decreases

    def summarize_trends(self, increases, decreases):
        # Setup OpenAI Client
        client = get_azure_openai_client()

        system_message = {
            "role": "system",
            "content": "You are a medical data analyst who summarizes clinical trial trends in plain English."
        }

        user_message = {
            "role": "user",
            "content": (
                f"Here are clinical trial activity changes by condition between {self.start_label} and {self.end_label}. "
                f"Please summarize the trend data below:\n\n"
                f"Notable increases:\n"
                "\n".join(increases) + 
                f"\n\nNotable decreases:\n" + "\n".join(decreases)
            )
        }

        functions = [
            {
                "name": "explain_condition_trends_increase_decrease",
                "description": "Generate a narrative explanation of the condition trends in clinical trials, and provide increases and decreases with numbers. (no percentages)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A plain-language summary of the most notable increases or decreases in trial activity."
                        },
                        "increases": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "5 most significant bullet points highlighting specific condition-level trends that go up. (display first months number and recent months number.)"
                        },
                        "decreases": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "5 most significant bullet points highlighting specific condition-level trends that go down. (display first months number and recent months number.)"
                        }
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
            temperature=0.1,
            max_tokens=2048
        )

        # Extract and return the structured response
        msg = response.choices[0].message
        if msg.function_call and msg.function_call.arguments:
            output = json.loads(msg.function_call.arguments)
            # Now return summary, increases, and decreases
            return output["summary"], output["increases"], output["decreases"]
        else:
            return "❌ GPT did not return a valid function call.", [], []



    def execute(self):
        # Fetch data and calculate changes
        increases, decreases = self.fetch_trend_data()

        # Generate highlights with numbers and percentages
        increases = self.generate_increases(increases)
        decreases = self.generate_decreases(decreases)

        # Summarize the trends using OpenAI
        summary, trend_increases, trend_decreases = self.summarize_trends(increases, decreases)


        # Return or print the result
        return summary, trend_increases, trend_decreases

# Example of usage
#agent = SummaryAgent(start_year=2012, start_month=4, end_year=2025, end_month=5)
#agent.execute()
