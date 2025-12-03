# Interactive House Price Explanation Dashboard using LIME

An interactive visualization system that explains house price predictions using LIME (Local Interpretable Model-agnostic Explanations).

**Course:** Interaction and Information Design  
**Team:** Urša Kumelj, Timen Bobnar, Matija Krigl  
**Institution:** University of Ljubljana, Faculty of Computer and Information Science

---

## Project Overview

This project combines machine learning with explainable AI to help users understand **what drives house prices**. Using the LIME algorithm, we provide transparent, interpretable explanations for individual price predictions.

### Key Features

- 🤖 **Random Forest Model** trained on 20,640 California housing samples
- 🔍 **LIME Explanations** showing top features impacting each prediction
- 📊 **Interactive Dashboard** built with Streamlit
- 🎛️ **What-If Analysis** - adjust features and see real-time price changes
- 🌍 **Multi-Market Support** - designed to work with both US and Slovenian data

---

## Quick Start

See **[SETUP.md](SETUP.md)** for detailed installation instructions.
```bash
# Clone and setup
git clone https://github.com/ursakumeljfaks/IOI_project.git
cd IOI_project
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model (first time only)
jupyter notebook  # Run notebooks/02_train_model.ipynb

# Launch dashboard
streamlit run streamlit_app/app.py
```

---

## Demo

![Dashboard Preview](docs/dashboard_preview.png) *(to be added)*

**Live Demo:** http://localhost:8501 (after running locally)

---

## Technology Stack

- **ML Framework:** scikit-learn (Random Forest Regressor)
- **Explainability:** LIME (lime-python)
- **Visualization:** Streamlit, Plotly
- **Data Processing:** pandas, numpy
- **Development:** Jupyter Notebooks

---

## Project Structure
```
house-price-lime/
├── data/                   # Datasets
├── models/                 # Trained models (not in git)
├── notebooks/              # Jupyter notebooks for development
├── src/                    # Core Python modules
├── streamlit_app/          # Interactive dashboard
├── docs/                   # Documentation & reports
├── requirements.txt
├── SETUP.md               # Detailed setup instructions
└── README.md
```

---

## Academic Context

**Based on:**
- Ribeiro et al., 2016: *"Why Should I Trust You?" Explaining the Predictions of Any Classifier*

**Course Requirements:**
- Research-based project extending existing approaches
- Interactive visualization using D3.js/Vega-Lite or similar
- Evaluation with user studies
- 4-page VGTC format paper
- 3-minute demo video

---


## License

This project is developed for academic purposes as part of the Interaction and Information Design course at University of Ljubljana.

---

## Acknowledgments

- California Housing Dataset from scikit-learn
- LIME library by Marco Tulio Ribeiro et al.
- Course instructors and teaching assistants

---

**For detailed setup instructions, see [SETUP.md](SETUP.md)**
