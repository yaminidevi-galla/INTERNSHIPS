# Intelligent Product Recommendation & Customer Purchase Pattern Analysis

This is a complete executable Python/Streamlit implementation of the uploaded IBM case study.

## Features

- Transaction data loading
- CSV upload
- Data preprocessing
- Duplicate and invalid quantity handling
- Binary transaction matrix
- Product frequency analysis
- Market Basket Analysis
- Apriori frequent itemset mining
- Association rule generation
- Support, confidence and lift
- Recommendation engine
- Graphs
- CSV downloads

## Project files

- `app.py` - complete application
- `transactions.csv` - sample transaction dataset
- `requirements.txt` - required Python libraries

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If `py` is not available, try:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application will open in your browser.

## CSV format

Your CSV should contain at least these two columns:

- `TransactionID`
- `ProductName`

Optional columns such as `Quantity`, `Date`, `CustomerID`, `UnitPrice`, and `Category` can also be included.

## How to demonstrate the project

1. Start the application.
2. Select `Case-study sample data`.
3. Open `Preprocessing` and show the binary transaction matrix.
4. Open `Purchase Patterns` and show product frequencies and frequent itemsets.
5. Open `Association Rules` and explain support, confidence and lift.
6. Open `Recommendation Engine`.
7. Select `Laptop`, `Smartphone`, or another product.
8. Explain that the system searches learned association rules and displays associated products.

The default threshold values in the application are implementation defaults and can be changed from the sidebar.
