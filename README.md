# ECOMMERCE-SALES-PREDICTION

![alt text](1_b4_2bCCwcmLtu-3tWN50IQ.jpg)

 Live App → http://10.254.0.91:8501

## 1. Business Understanding
Accurately forecast daily unit-sales for the next 90 days so that product managers and supply-chain teams can:
- minimize stock-outs & over-stock,
- optimize marketing spend,
- set dynamic pricing, and
- forecast revenue.

**Success Metrics**
- Forecast Accuracy: RMSE ≤ 4.0 and MAPE ≤ 12 %.
- Business uptake: ≥ 80 % of stakeholders adopt the Streamlit dashboard for weekly planning.

## 2. Data Understanding
| Variable           | Type        | Example     | Business Meaning |
| ------------------ | ----------- | ----------- | ---------------- |
| `Date`             | Date        | 2023-01-01  | Transaction day  |
| `Product_Category` | Categorical | Electronics | SKU group        |
| `Price`            | Numeric     | 599.99      | Selling price    |
| `Discount`         | Numeric     | 40.00       | Absolute coupon  |
| `Customer_Segment` | Categorical | Premium     | Loyalty tier     |
| `Marketing_Spend`  | Numeric     | 3 500       | Daily ad budget  |
| `Units_Sold`       | Target      | 32          | Units sold       |

**Quick Stats**
- 1 000 rows, 2023-01-01 to 2025-09-03 (daily).
- Zero missing values, 5 categories, 3 customer segments.
- Outliers handled via IQR capping.

## 3.Data Preparation
#### Cleaning
- Date parsing & sort.
- Outlier capping (Units_Sold within [Q1-1.5IQR, Q3+1.5IQR]).
- Non-negative constraints.
#### Feature Engineering
Lag features: Units_Sold_d-1, d-7.
Rolling means: 7-day, 14-day.
Cyclical time: DayOfWeek_sin, DayOfWeek_cos.
Price elasticity: Price / Discount.
One-hot encoding for categorical variables.

## 4. Modeling
| Model                 | Hyper-parameter Search                               | RMSE (CV) | Notes                     |
| --------------------- | ---------------------------------------------------- | --------- | ------------------------- |
| **Linear Regression** | 5-fold TimeSeriesSplit                               | 5.9       | Baseline, interpretable   |
| **Random Forest**     | GridSearch (`n_estimators`, `max_depth`)             | 4.3       | Captures non-linearities  |
| **XGBoost**           | RandomizedSearchCV (`eta`, `max_depth`, `subsample`) | **3.8**   | Best performer            |
| **Prophet**           | Cross-validation horizon = 30 days                   | 4.1       | Built-in holidays & trend |

## 5. Evaluation
| Metric | XGBoost | Random-Forest | Prophet | Linear |
| ------ | ------- | ------------- | ------- | ------ |
| RMSE   | 3.81    | 4.29          | 4.15    | 5.94   |
| MAE    | 3.02    | 3.39          | 3.25    | 4.78   |
| MAPE   | 11.4 %  | 12.8 %        | 12.2 %  | 18.7 % |

**Business Sanity Check**
Forecast tracks promotions & holiday spikes.
Feature importance: Price, Discount, Marketing_Spend dominate.

## 6. Deployment

#### Streamlit Dashboard

1. Upload CSV → instant preview.
2. Model selector → side-by-side forecast plots.
3. Scenario slider: “+20 % marketing spend” recalculates forecast live.
4. Export button → CSV of 90-day predictions.
5. Auto-retraining nightly via GitHub-Actions.

#### Local Run

git clone https://github.com/your-username/ecommerce-sales-predictor.git

cd ecommerce-sales-predictor

pip install -r requirements.txt

streamlit run app.py

####  Repository Structure

├── 01-Business-Understanding/

│   └── README.md

├── 02-Data-Understanding/

│   └── EDA.ipynb

├── 03-Data-Preparation/

│   └── prep.py

├── 04-Modeling/

│   ├── train.py

│   └── notebooks/

├── 05-Evaluation/

│   └── metrics.json

├── 06-Deployment/

│   └── app.py

├── models/       

├── assets/ 

└── requirements.txt

## 7. Limitations 

Data Scope: The dataset covers only 1,000 records, which may not fully represent long-term sales patterns or seasonality effects.

Feature Coverage: Important external drivers such as holidays, competitor activity, or economic conditions are missing, limiting the model’s ability to capture real-world demand influences.

Model Performance: The Random Forest model showed weak predictive power (negative R²), indicating it struggles to explain sales variation.

Residual Patterns: Residuals are not randomly distributed, suggesting the model has bias and may underperform on unseen data.

Data Granularity: Data is aggregated at daily product-category level but does not account for store-level or regional variations, which could impact accuracy.

### License

MIT © 2024 Loise-Kabogo

