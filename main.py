import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

# Load dataset
data = pd.read_excel('dataIN/AGRIECO.xlsx', sheet_name='VANZARI', index_col=0)

# Custom Styling
st.markdown(
    """
    <style>
    .title{
        color:#1E3A8A !important;
    }
    .custom-title {
        color: #26822d !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">AGRIECO Sales Analytics</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="custom-title">Enache Petru & Dușa Andreea</h3>', unsafe_allow_html=True)

# Navigation sidebar
section = st.sidebar.radio("Jump to section:",
                           ["Dataset Overview", "DataFrame Exploration", "Outlier Analysis",
                            "Data Aggregations", "Histograms", "Correlations & Distributions", "Data Preprocessing"])

if section == "Dataset Overview":
    st.header("Agrico Dataset Analysis")
    st.markdown("""
    # 🏬 About *Agrico*
    Agrico is a key player in the agri-food sector, focusing on the production and distribution of fresh produce. Backed by a wide network of local growers and partners, the company supplies high-quality goods to both local and international markets. 

    ## 🌿 Quick Facts
    - 🌍 Partners directly with local farmers to guarantee fresh and sustainable produce.
    - 💧 Employs efficient irrigation techniques to conserve natural resources.
    - 🍃 Committed to eco-friendly farming practices and minimizing food waste.

    ---

    # 📊 About the Dataset
    This dataset records delivery and sales data for Agrico products, capturing product categories, quantities, pricing structures, and transaction timestamps.

    ## 🔍 Variables & Features
    - **🌽 Product Name:** The specific item name (e.g., Broccoli 1KG, Beans 450G).
    - **📦 Quantity:** Number of units delivered per transaction.
    - **💰 Unit Price:** Cost per individual unit.
    - **⚖️ Unit Weight:** Weight of a single unit (expressed in kg).
    - **📅 Delivery Date:** The date the order was dispatched to clients.
    - **🆔 Category Code:** Unique numerical identifier for product groupings.
    - **🥦 Category Name:** High-level product classification (e.g., VEGETABLES).
    - **🎯 Discount Value:** Financial reduction applied to the final price.
    - **⚖️ Total Weight:** Cumulative weight calculated from units and individual item weight.

    ---

    # 📈 Key Sales Insights
    ## 📊 Core Observations
    - **🥇 Top Movers:** Broccoli, Peas, and Beans consistently rank as the most frequently shipped vegetables.
    - **📆 Delivery Timing:** Shipment volumes peak heavily during the first half of each month, pointing to specific demand cycles.
    - **💸 Discounting Strategy:** Price cuts are primarily leveraged to move high-volume inventory (e.g., Cauliflower 450G).
    - **⚖️ Volume Drivers:** Peas 1KG and Beans 1KG account for the highest aggregate weights shipped, validating strong market demand.
    - **🥗 Premium Tier:** Items like Beans and Broccoli command higher price points, forming a distinct premium segment.

    ---

    # 🚀 Strategic Recommendations
    - ✅ **Optimize Fulfillment Waves:** Boost supply chain capacity in the second half of the month to balance inventory flow.
    - ✅ **Expand Premium Lines:** Scale up inventory for high-demand items like Broccoli and Beans to capture higher margins.
    - ✅ **Targeted Promotions:** Use dynamic discounts on slower-moving items (e.g., Cauliflower 450G) to clear stock efficiently.
    - ✅ **Diversify Sourcing:** Broaden the supplier network to stay competitive across seasonal shifts.
    - ✅ **Seasonality Modeling:** Tailor production schedules closely to consumer buying trends to maximize profitability.

    ---

    """)

elif section == "DataFrame Exploration":
    st.header('Raw Dataset')
    st.dataframe(data)
    
    data['data_livrare'] = pd.to_datetime(data['data_livrare'])
    data_grouped = data.groupby('data_livrare')['cantitate_um'].sum().reset_index()

    st.header("📈 Sales Timeline Trend")

    fig, ax = plt.subplots()
    ax.plot(data_grouped['data_livrare'], data_grouped['cantitate_um'], marker='o', linestyle='-', color='#1E3A8A')
    ax.set_xlabel("Delivery Date")
    ax.set_ylabel("Quantity Sold")
    ax.set_title("Agrico Product Sales Progression Over Time")
    ax.grid(True)

    st.pyplot(fig)

    # Missing values check
    st.subheader("📌 Missing Values")
    st.write(data.isnull().sum())

    # Duplicate check
    st.subheader("🔄 Duplicate Records")
    st.write(f"Number of duplicate rows: {data.duplicated().sum()}")

elif section == "Outlier Analysis":
    st.markdown("""
    # 📊 Outlier Detection

    ## 🔍 What are outliers?
    Outliers are data points that fall abnormally far from the general distribution. If left unchecked, they can skew metrics like the mean and standard deviation.

    ## 📌 The Interquartile Range (IQR) Method
    To flag anomalies cleanly, we rely on the IQR formula:
    - **Q1 (First Quartile)**: 25% of data falls below this threshold.
    - **Q3 (Third Quartile)**: 75% of data falls below this threshold.
    - **IQR**: The spread between Q3 and Q1.
    - **Lower Bound**: Q1 - (1.5 * IQR)
    - **Upper Bound**: Q3 + (1.5 * IQR)

    Anything outside these safety boundaries is marked as an outlier.
    """, unsafe_allow_html=True)

    st.subheader("⚠️ Outlier Breakdown by Column")
    outlier_cols = []
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            outlier_cols.append(col)
            st.write(f"🔍 Column `{col}` contains {outliers.shape[0]} outlier entries.")

    if not outlier_cols:
        st.write("✅ No outliers detected in this dataset.")

    # Boxplot visualization
    st.subheader("📊 Boxplot Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data.select_dtypes(include=[np.number]), ax=ax)
    ax.set_title("Feature Boxplots for Extreme Value Detection")
    st.pyplot(fig)

    st.subheader("📋 Dataset Shape")
    st.write(f"📌 Total rows: {data.shape[0]}")
    st.write(f"📌 Total columns: {data.shape[1]}")

    st.subheader("📊 Descriptive Statistics")
    st.write(data.describe())

elif section == "Data Aggregations":
    data["data_livrare"] = pd.to_datetime(data["data_livrare"], format="%m/%d/%Y")

    st.markdown("""
    # 📊 Aggregated Metrics
    """, unsafe_allow_html=True)

    total_cantitate = data.groupby("denumire_produs")["cantitate_um"].sum()
    total_venituri = data.groupby("denumire_produs").apply(lambda x: (x["cantitate_um"] * x["pret"]).sum())
    total_discount = data.groupby("denumire_produs")["valoare_discount"].sum()
    evolutie_vanzari = data.groupby("data_livrare")["cantitate_um"].sum()

    st.subheader("📊 Total Quantity Sold per Product")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(total_cantitate.index, total_cantitate.values, color='blue')
    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Product")
    ax.set_title("Total Volume per Product")
    st.pyplot(fig)

    st.subheader("📊 Total Revenue per Product")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(total_venituri.index, total_venituri.values, color='green')
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Product")
    ax.set_title("Revenue Contribution by Product")
    st.pyplot(fig)

    st.subheader("📊 Total Discounts Applied per Product")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(total_discount.index, total_discount.values, color='red')
    ax.set_xlabel("Discount Value")
    ax.set_ylabel("Product")
    ax.set_title("Discounts Issued per Product")
    st.pyplot(fig)

    st.subheader("📈 Sales Trajectory Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evolutie_vanzari.index, evolutie_vanzari.values, marker='o', linestyle='-', color='purple')
    ax.set_xlabel("Delivery Date")
    ax.set_ylabel("Quantity Sold")
    ax.set_title("Daily Sales Progression")
    ax.grid(True)
    st.pyplot(fig)

elif section == "Histograme":
    data["data_livrare"] = pd.to_datetime(data["data_livrare"], format="%m/%d/%Y")

    st.markdown("""
    # 📊 Distribution Histograms
    """, unsafe_allow_html=True)

    st.subheader("📊 Quantity Sold Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["cantitate_um"], bins=20, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram - Quantity Sold Spread")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    The distribution is heavily right-skewed, meaning the vast majority of orders involve smaller quantities (under 500 units). Rare large orders exceeding 1,000 units drive the tail end of the distribution.
    """, unsafe_allow_html=True)

    st.subheader("📊 Unit Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["pret"], bins=20, color='green', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Unit Price")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram - Unit Price Spread")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    Prices cluster into distinct bands, reflecting structured pricing tiers. Most items sit between 6 and 15 RON per unit, while higher-priced specialty goods appear less frequently.
    """, unsafe_allow_html=True)

    st.subheader("📊 Discount Value Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["valoare_discount"], bins=20, color='red', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Discount Amount")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram - Discount Value Spread")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    Most discounts are kept quite small (under 200 RON). This long-tail pattern indicates that standard transactions carry minor promotional cuts, while heavy discounts are reserved for clearance or special campaigns.
    """, unsafe_allow_html=True)

    st.subheader("📊 Total Weight Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["greutate_kg"], bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Total Weight Shipped")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram - Total Weight Spread")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    Mirrors the quantity distribution closely—crowded around lower weights (under 500 kg) with sharp spikes pointing to bulk corporate or enterprise deliveries.
    """, unsafe_allow_html=True)

elif section == "Corelații și Distribuții":
    st.markdown("""
    # 🔍 Correlations & Distributions
    Exploring feature relationships and categorical spreads across the portfolio.
    """, unsafe_allow_html=True)

    st.subheader("🧠 Correlation Matrix")
    numeric_cols = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_cols.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    There is a very robust correlation between `greutate_kg` and `cantitate_um` (**0.87**), as well as `greutate_kg` and `valoare_discount` (**0.93**)—which makes intuitive sense since larger orders increase both total weight and discount leverage. Additionally, `pret` tracks closely with `greutate_um` (**0.73**), indicating heavier items carry higher base prices.
    """, unsafe_allow_html=True)

    st.subheader("📦 Volume by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='denumire_categorie', data=data, order=data['denumire_categorie'].value_counts().index,
                  palette='viridis', ax=ax)
    ax.set_title("Shipment Count by Product Category")
    ax.set_xlabel("Count")
    ax.set_ylabel("Category")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    **VEGETABLES** lead total volume by a wide margin, followed by **FRUITS** and **MIXED VEGETABLES**, confirming Agrico's core focus on fresh plant produce.
    """, unsafe_allow_html=True)

    st.subheader("📦 Top Products by Frequency")
    top_products = data['denumire_produs'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='magma', ax=ax)
    ax.set_title("Top 10 Shipped Products")
    ax.set_xlabel("Order Frequency")
    ax.set_ylabel("Product")
    st.pyplot(fig)
    st.markdown("""
    ✅ **Takeaway:**
    Mid-sized packages like **Peas 450G**, **Broccoli 450G**, and **Raspberries 450G** dominate customer preferences, while mixed blends point to steady demand from modern retail and food service channels.
    """, unsafe_allow_html=True)

elif section == "Preprocesare Date":
    st.markdown("## 🧪 Data Preprocessing - Encoding & Scaling", unsafe_allow_html=True)

    st.subheader("🔤 One-Hot Encoding Categories")
    encoded_data = pd.get_dummies(data, columns=["denumire_categorie"], drop_first=True)
    st.write(encoded_data.head())
    st.markdown("""
    ✅ **Note:** Categorical text fields are converted into binary columns to prevent models from assuming a false ordinal ranking between categories.
    """, unsafe_allow_html=True)

    st.subheader("🔢 Label Encoding (`denumire_produs` example)")
    label_encoder = LabelEncoder()
    encoded_data["denumire_produs_encoded"] = label_encoder.fit_transform(data["denumire_produs"])
    st.write(encoded_data[["denumire_produs", "denumire_produs_encoded"]].head())
    st.markdown("""
    ✅ **Note:** Maps text names to numeric indices—handful for tree-based architectures or categorical identification.
    """, unsafe_allow_html=True)

    st.subheader("📏 Standardization (StandardScaler)")
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    scaled_data_std = data.copy()
    scaled_data_std[numeric_cols] = scaler.fit_transform(scaled_data_std[numeric_cols])
    st.write(scaled_data_std[numeric_cols].head())
    st.markdown("""
    ✅ **Note:** Centers features around a mean of 0 with a standard deviation of 1. Ideal for regression models or principal component analysis.
    """, unsafe_allow_html=True)

    st.subheader("📊 Normalization (MinMaxScaler)")
    minmax = MinMaxScaler()
    scaled_data_minmax = data.copy()
    scaled_data_minmax[numeric_cols] = minmax.fit_transform(scaled_data_minmax[numeric_cols])
    st.write(scaled_data_minmax[numeric_cols].head())
    st.markdown("""
    ✅ **Note:** Squeezes all values neatly into a bounded `[0, 1]` window while keeping the underlying distribution shape intact.
    """, unsafe_allow_html=True)

    st.subheader("📊 Distribution Comparison - Original vs Standardized vs MinMax")
    feature_to_compare = "pret"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[feature_to_compare], bins=20, color='blue', label='Original', kde=True)
    sns.histplot(scaled_data_std[feature_to_compare], bins=20, color='green', label='Standardized', kde=True)
    sns.histplot(scaled_data_minmax[feature_to_compare], bins=20, color='red', label='Normalized', kde=True)
    ax.set_title(f"Distribution Comparison for '{feature_to_compare}'")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"""
    ✅ **Takeaway:**
    - **Blue**: Original raw spread.
    - **Green**: Scaled to zero-mean and unit variance.
    - **Red**: Bounded strictly between 0 and 1.
    
    This visual check ensures transformations behave correctly prior to feeding data into machine learning algorithms.
    """, unsafe_allow_html=True)
