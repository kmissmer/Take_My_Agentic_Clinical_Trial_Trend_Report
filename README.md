# ⚖️ AI Legal Assistant

This project builds an AI-powered assistant that helps users understand U.S. legal opinions. It uses public court data from the CourtListener API and OpenAI models to summarize court decisions, explain legal terms, and surface similar cases.

## 🔍 Features

- 📄 **Case Summarizer Agent** – Converts dense legal opinions into plain-language summaries.
- 📚 **Legal Term Explainer** – Identifies and explains complex legal concepts in each case.
- 🔁 **Similar Case Finder** – Uses sentence embeddings to find cases with similar fact patterns or rulings.
- 🏛️ **Court Metadata Annotator** – Adds helpful context such as court level, location, and date filed.

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/ai-legal-assistant.git
cd ai-legal-assistant
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Get API Keys

#### 🏛️ CourtListener API
- Sign up at [https://www.courtlistener.com/](https://www.courtlistener.com/)
- Navigate to your account page to find your API token

#### 🤖 OpenAI API (for GPT)
- Get an API key for this project
- Save it in a `.env` file:

```env
COURTLISTENER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 4. Run a Demo Script

```bash
python scripts/fetch_and_summarize.py
```

---

## 🧠 Project Structure

```
├── data/                      # Optional: cache downloaded cases
├── scripts/
│   ├── fetch_and_summarize.py  # Downloads opinions & generates summaries
│   └── similar_cases.py        # Uses embeddings to find similar opinions
├── agents/
│   ├── summarizer.py           # GPT-based summarizer
│   └── explainer.py            # Legal concept explainer
├── utils/
│   └── api_client.py           # Handles API requests
├── README.md
└── requirements.txt
```

---

## 🔧 Future Work

- Build a Streamlit web UI for interactive exploration
- Add clustering to group cases by theme
- Train a fine-tuned summarizer on long legal texts

---

## 📚 References

- [CourtListener API Docs](https://www.courtlistener.com/api/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Free Law Project](https://free.law/)

---

## 💡 License

This project is open-source under the MIT License. Please ensure your usage complies with the CourtListener and OpenAI terms of service.
