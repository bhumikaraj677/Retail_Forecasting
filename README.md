# Retail Sales Forecasting Using Machine Learning

**An end-to-end retail sales analysis and demand forecasting project**

**Author:** Bhumika Raj

## Project Overview

This project analyzes retail transaction data, explores sales trends, compares forecasting models, and predicts sales for the next seven days. It includes a Streamlit application for interactive analysis and Docker support for local deployment.

## Features

- Retail transaction data analysis
- Daily, monthly, and product-category sales summaries
- Sales trend visualizations
- Machine learning-based sales forecasting
- Forecasting model comparison
- Model evaluation using MAE, RMSE, and R-squared
- Seven-day sales forecast
- Interactive Streamlit dashboard
- Docker-based application deployment

## Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Web application | Streamlit |
| Containerization | Docker |
| Version control | Git and GitHub |

## Project Structure

```text
Retail_Forecasting/
├── app.py
├── main.py
├── dockerfile
├── requirements.txt
├── retail_sales_dataset.csv
├── outputs/
└── README.md
```

## Dataset

The project uses `retail_sales_dataset.csv`, which contains transaction details such as transaction dates, customer identifiers, product categories, quantities, unit prices, and total amounts.

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/bhumikaraj677/Retail_Forecasting.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Retail_Forecasting
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Run with Docker

Build the Docker image:

```bash
docker build -t retail-forecasting -f dockerfile .
```

Run the container:

```bash
docker run --rm -p 8501:8501 retail-forecasting
```

Open `http://localhost:8501` in your browser.

## Model Evaluation

Forecasting performance is assessed using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R-squared:** A measure of model fit

Refer to the generated files in the `outputs/` directory for the available evaluation results and forecast visualizations. Actual performance should be judged from the saved metrics, not assumed from the project description.

## Live Demo

[Open the Retail Forecasting Streamlit App](https://retailforecasting-ihxdhkptpxsdcp3mwrupto.streamlit.app/)

## Future Improvements

- Customer segmentation using RFM analysis and clustering
- Customer churn prediction
- More advanced demand forecasting experiments
- Inventory reorder recommendations
- Model tracking and drift monitoring
- Automated testing and continuous integration

These are planned enhancements and should be marked complete only after implementation and testing.

## Author

**Bhumika Raj**

- GitHub: https://github.com/bhumikaraj677

## Project Status

The project includes a retail forecasting foundation and a Streamlit application. Further analytics, validation, and production-readiness improvements are planned as part of the RetailPulse roadmap.
