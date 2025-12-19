# House Price LIME Dashboard - Setup Instructions

**Project:** Interactive House Price Explanation using LIME  
**Team:** Urša Kumelj, Timen Bobnar, Matija Krigl  
**Course:** Interaction and Information Design

---

## Prerequisites

- **Python 3.11** (pomembno! 3.14 ne dela zaradi PyArrow)
- **Git** installed
- **GitHub account** with access to this repo

---

##  Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/ursakumeljfaks/IOI_project.git
cd IOI_project
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- pandas, numpy, scikit-learn (data & ML)
- lime (explanations)
- streamlit, plotly (visualization)
- jupyter (notebooks)

**Installation time:** ~5-10 minutes

---

## Running the Project

### Option A: Run Streamlit Dashboard (Interactive UI)
```bash
# Make sure venv is activated
streamlit run streamlit_app/app.py
```

**Opens automatically in browser:** http://localhost:8501

**What you'll see:**
- Sliders on left sidebar to adjust house features
- Predicted price in real-time
- LIME explanation chart (green/red bars)
- What-if analysis at bottom

**To stop:** Press `Ctrl+C` in terminal

---

### Option B: Run Jupyter Notebooks (Development)
```bash
# Make sure venv is activated
jupyter notebook
```

**Opens Jupyter in browser** at http://localhost:8888

**Key notebooks:**
1. `notebooks/01_data_acquisition.ipynb` - Load California Housing dataset
2. `notebooks/02_train_model.ipynb` - Train Random Forest model with LIME

**Run notebooks:** Click "Cell → Run All" or use the ▶▶ button

---

## Project Structure
```
house-price-lime/
├── data/
│   ├── geeksforgeeks/          # California Housing dataset (20k samples)
│   └── slovenian/              # Slovenian properties (TO BE ADDED)
├── models/
│   └── house_price_model.pkl   # Trained Random Forest (NOT in git - too large)
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   └── 02_train_model.ipynb
├── src/
│   ├── model.py                # HousePriceModel class
│   └── explainer.py            # LIMEExplainer class
├── streamlit_app/
│   └── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## First Time Setup - Train the Model

**Important:** The trained model (`house_price_model.pkl`) is NOT in Git because it's 34MB.  
You need to train it once:
```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start Jupyter
jupyter notebook

# In browser:
# 1. Open and run: notebooks/01_data_acquisition.ipynb
# 2. Open and run: notebooks/02_train_model.ipynb
```

**Expected results:**
- RMSE: ~0.5 (meaning ±$50k error)
- R² score: ~0.80-0.85
- Model saved to: `models/house_price_model.pkl`

**Training time:** ~2-3 minutes

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:** Make sure you're running from project root directory:
```bash
cd ~/IOI_project  # or wherever you cloned it
streamlit run streamlit_app/app.py
```

### Issue: `Model not found!` in Streamlit
**Solution:** You need to train the model first (see "First Time Setup" above)

### Issue: `pip install` fails for pandas/pyarrow
**Solution:** Use Python 3.11 (NOT 3.14):
```bash
# Check version
python --version

# If wrong version, install Python 3.11:
# macOS: brew install python@3.11
# Windows: Download from python.org
```

### Issue: Port 8501 already in use
**Solution:** Kill existing Streamlit process:
```bash
# Find process
lsof -ti:8501

# Kill it
kill -9 $(lsof -ti:8501)

# Or use different port
streamlit run streamlit_app/app.py --server.port 8502
```

---

## Testing the Setup

Run this quick test to verify everything works:
```python
# In Python terminal (with venv activated):
python << 'PYTEST'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lime
import streamlit
import plotly

print("All packages imported successfully!")
print(f"Python: {__import__('sys').version}")
print(f"pandas: {pd.__version__}")
print(f"streamlit: {streamlit.__version__}")
print(f"lime: {lime.__version__}")
PYTEST
```

---

## Getting Help

**If you encounter issues:**

1. **Check this SETUP.md** for common solutions
2. **Message the team** in our WhatsApp/Messenger group
3. **GitHub Issues** - open an issue on the repo
4. **Office hours** with professor

---

## Next Steps

Once you have everything running:

1. **Explore the Streamlit dashboard** - move sliders, see how LIME explanations change
2. **Review the notebooks** - understand the data and model training process
3. **Read the code** - `src/model.py` and `src/explainer.py`
4. **Start collecting Slovenian data** - see `data/slovenian/template.csv`

---

## Useful Resources

- **LIME Paper:** Ribeiro et al., 2016 - "Why Should I Trust You?"
- **Streamlit Docs:** https://docs.streamlit.io
- **California Housing Dataset:** https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- **Project Requirements:** See `Interaction_and_Information_Design_-_Project.pdf`

---

**Questions? Issues? Reach out to the team!** 