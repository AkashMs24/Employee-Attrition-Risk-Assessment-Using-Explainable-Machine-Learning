<div align="center">

# Employee Attrition Risk Assessment

### Explainable Machine Learning for HR Decision-Making

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://explainable-hr-attrition-risk-system.streamlit.app/)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square)
![XAI](https://img.shields.io/badge/XAI-SHAP-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)

</div>

---

## The problem

Every time a company loses an employee, it costs **50–200% of their annual salary** to replace them. HR teams often find out too late — after resignation, not before.

Most attrition tools give a binary answer: *will leave* or *won't leave*.  
That's not useful. HR needs to know **who is at risk, how much risk, and exactly why.**

---

## What this system does differently

This is not a classifier. It's a **decision-support system.**

- Outputs a **probability score** (0–100% risk), not just yes/no
- Explains *why* each employee is flagged — using SHAP feature importance
- Lets HR simulate interventions — *"what happens if we give a 10% raise?"*
- Processes entire teams via CSV batch upload
- Generates downloadable prediction reports for HR review

---

## Example output
Employee: John D.
Attrition Risk: 78%  ⚠️ High
Top risk factors:
↑ Overtime hours        (+23% risk contribution)
↑ Years since promotion (+18% risk contribution)
↓ Job satisfaction      (+15% risk contribution)
What-If: +10% salary increase → Risk drops to 61%

---

## Features

| Feature | What it enables |
|---|---|
| Probability risk score | Flexible thresholds — HR sets their own alert level |
| SHAP explainability | Every prediction justified, not just flagged |
| What-If salary simulation | Test interventions before implementing them |
| Batch CSV prediction | Score entire departments at once |
| Downloadable report | Ready for HR review meetings |

---

## Model

- **Algorithm:** Random Forest Classifier
- **Output:** Attrition probability (0.0 – 1.0)
- **Features:** Overtime, job satisfaction, years since promotion, salary band, tenure, and more
- **Explainability:** SHAP values per prediction
- **Deployment:** Lightweight inference pipeline — training and deployment separated for speed

---

## Stack

`Python` `scikit-learn` `SHAP` `Streamlit` `Pandas` `NumPy` `Joblib`

---

## Run locally

```bash
git clone https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning.git
cd Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning
pip install -r requirements.txt
streamlit run app.py
```

---

## Design decisions worth noting

- **Probability over binary** — a 78% risk score is actionable; "will leave" is not
- **SHAP over black-box** — HR can challenge and verify every recommendation
- **Simulation over prediction** — what-if analysis turns insight into intervention
- **Separated pipelines** — model trained offline, deployed artifact is lightweight for fast inference

---

## Related projects

- [Decision Intelligence System](https://github.com/AkashMs24/Decisioniq-ai-business-intelligence) — ML + LLM business intelligence platform
- [Fraud Detection System](https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System) — XGBoost + SHAP + FastAPI
- [FarmVoice AI](https://github.com/AkashMs24/FarmVoice-AI) — NLP + SHAP crop advisory

---

<div align="center">

Built by **Akash M S** · Presidency University, Bengaluru  
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com

</div>
