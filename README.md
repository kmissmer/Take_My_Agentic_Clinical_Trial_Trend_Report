# 🧠 Take_My_Agentic_Clinical_Trial_Trend_Report

This project uses AI agents to analyze clinical trial activity in the public AACT database and generate plain-language trend reports. Built for learning and demonstrating how AI agents can perform coordinated SQL querying, data interpretation, and natural language summarization.

## 🎯 What It Does

- 📊 **Trend Detector Agent**: Runs SQL on the AACT database to detect changes in trial activity (e.g., new trials, top conditions, locations).
- 🧠 **Insight Generator Agent**: Uses an LLM to analyze the trends and generate meaningful insights (e.g., spikes, emerging conditions).
- 📝 **Report Writer Agent**: Creates a human-readable report summarizing the month’s trends, formatted in Markdown.

## 🧪 Example Output

```
## 🔬 Clinical Trial Trends – April 2025

- 📈 **Top Growing Conditions**:
  - Alzheimer’s: +23%
  - Rare Cancers: +15%

- 🌍 **Most Active States**:
  - California (130 new trials)
  - Texas (118 new trials)

- 🧪 **Notable Observations**:
  - Increased activity in mRNA and gene therapy trials.
  - Multiple sponsors targeting pediatric conditions.

- 🧾 **Overall**: 6,284 new trials registered — a 12% increase from March.
```

## 🧱 Architecture

```
├── app.py                           # Optional Streamlit or CLI entrypoint
├── agent_coordinator.py            # Orchestrates all agent behavior
├── agents/
│   ├── trend_detector.py           # SQL-powered trends agent
│   ├── insight_generator.py        # GPT-powered insight agent
│   └── report_writer.py            # Markdown writer agent
├── utils/
│   └── sql_util.py                 # Helper to query the AACT Postgres DB
├── reports/
│   └── latest_report.md            # Output file
```

## 🚀 Getting Started

1. Clone the repo:
```bash
git clone https://github.com/yourusername/Take_My_Agentic_Clinical_Trial_Trend_Report.git
cd Take_My_Agentic_Clinical_Trial_Trend_Report
```

2. Set up environment:
```bash
pip install -r requirements.txt
```

3. Configure your `.env`:
```
OPENAI_API_KEY=your-key
PG_URI=postgresql://user:pass@localhost:5432/aact
```

4. Run the trend report:
```bash
python app.py
```

## 🧠 Built With

- 🗃️ AACT Clinical Trial Database (Postgres)
- 🤖 OpenAI GPT-4o or GPT-3.5 for summarization
- 🐍 Python (with SQL, Pandas, and LangChain/OpenAI function-calling style)

## 📚 References

- [AACT Schema](https://aact.ctti-clinicaltrials.org/schema)
- [OpenAI API](https://platform.openai.com/docs)
- [CTTI/ClinicalTrials.gov](https://clinicaltrials.gov/)

## 🪪 License

MIT — open to use and extend for research, personal learning, and portfolio projects.
