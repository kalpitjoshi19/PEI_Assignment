# Databricks notebook source
# DBTITLE 1,Instal required libraries
#%pip install pytest
#%pip install openpyxl

# COMMAND ----------

# DBTITLE 1,ETL, Data and Schema check TCs
# ======================================================
#     test_etl_pipeline.py  (FULL END-TO-END COVERAGE)
# ======================================================

import pytest
from pyspark.sql import Row
from pyspark.sql.functions import col, size
from datetime import datetime
import pandas as pd
import os
import sys

# Import ETL functions
#from etl import (
#    read_customers, read_products, read_orders,
#    clean_customers, clean_products, clean_orders,
#    build_customer_dim, build_product_dim, build_orders_enriched,
#    aggregate_profit_by
#)

# ======================================================
#                PYSPARK FIXTURE
# ======================================================

@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder.master("local[2]")
        .appName("ETLTestFull")
        .getOrCreate()
    )


# ======================================================
#               SAMPLE DATA FIXTURES
# ======================================================

@pytest.fixture
def customers_raw(spark):
    return spark.createDataFrame([
        Row(**{"Customer ID": "C1", "Customer Name": "John Doe", "email": "john@example.com", 
               "phone": "1234567890", "Postal Code": "12345", "Region": "North", "Country": "USA"}),

        Row(**{"Customer ID": "C2", "Customer Name": None, "email": "jane@example.com", 
               "phone": "1234567890", "Postal Code": "12345", "Region": "South", "Country": "USA"}),

        Row(**{"Customer ID": "C3", "Customer Name": "Invalid@Name", "email": "invalidemail", 
               "phone": "notaphone", "Postal Code": "ABCDE", "Region": "Unknown", "Country": "USA"}),

        Row(**{"Customer ID": "C4", "Customer Name": "   ", "email": "empty@example.com", 
               "phone": "1234567890", "Postal Code": "12345", "Region": "East", "Country": "USA"}),

        # duplicate ID
        Row(**{"Customer ID": "C1", "Customer Name": "Duplicate", "email": "dup@example.com", 
               "phone": "1234567890", "Postal Code": "12345", "Region": "West", "Country": "USA"}),

        # missing ID
        Row(**{"Customer ID": None, "Customer Name": "NoID", "email": "noid@example.com", 
               "phone": "1234567890", "Postal Code": "12345", "Region": "West", "Country": "USA"}),
    ])

@pytest.fixture
def products_raw(spark):
    return spark.createDataFrame([
        Row(**{
            "Product ID": "P1", "Product Name": "Widget", "UnitPrice": 10.0,
            "Category": "Office Supplies", "Sub-Category": "Binders"
        }),
        Row(**{
            "Product ID": "P2", "Product Name": "Gadget", "UnitPrice": -5.0,
            "Category": "Technology", "Sub-Category": "Phones"
        }),
        Row(**{
            "Product ID": "P1", "Product Name": "Duplicate", "UnitPrice": 15.0,
            "Category": "Office Supplies", "Sub-Category": "Binders"
        }),
        Row(**{
            "Product ID": None, "Product Name": "NoID", "UnitPrice": 15.0,
            "Category": "Furniture", "Sub-Category": "Chairs"
        }),
    ])

@pytest.fixture
def orders_raw(spark):
    return spark.createDataFrame([
        Row(**{"Order ID": "O1", "Customer ID": "C1", "Product ID": "P1", 
               "Profit": 5.0, "Order Date": "1/1/2023", "Ship Date": "2/1/2023"}),

        Row(**{"Order ID": "O2", "Customer ID": "C2", "Product ID": "P2", 
               "Profit": -2.0, "Order Date": "1/2/2023", "Ship Date": "2/2/2023"}),

        # invalid IDs
        Row(**{"Order ID": "O3", "Customer ID": "C5", "Product ID": "P3", 
               "Profit": 3.0, "Order Date": "1/3/2023", "Ship Date": "2/3/2023"}),

        # Missing required ID
        Row(**{"Order ID": None, "Customer ID": "C1", "Product ID": "P1", 
               "Profit": 5.0, "Order Date": "1/4/2023", "Ship Date": "2/4/2023"}),

        # Duplicate order
        Row(**{"Order ID": "O1", "Customer ID": "C1", "Product ID": "P1", 
               "Profit": 5.0, "Order Date": "1/1/2023", "Ship Date": "2/1/2023"}),
    ])


# ======================================================
#           TEST — READ FUNCTIONS
# ======================================================

def test_read_customers(tmp_path, spark):
    df = pd.DataFrame({
        "Customer ID": ["A", "B"],
        "Customer Name": ["X", "Y"],
        "email": ["x@test.com", "y@test.com"],
        "phone": ["1234567890", "9876543210"],
        "Postal Code": ["11111", "22222"],
        "Region": ["North", "South"]
    })

    file_path = tmp_path / "customers.xlsx"
    df.to_excel(file_path, index=False)

    spark_df = read_customers(spark, str(file_path))
    assert spark_df.count() == 2
    assert "Customer ID" in spark_df.columns


def test_read_products(tmp_path, spark):
    df = pd.DataFrame({
        "Product ID": ["P1"],
        "Price": [20]
    })
    path = tmp_path / "products.csv"
    df.to_csv(path, index=False)

    spark_df = read_products(spark, str(path))
    assert spark_df.count() == 1


def test_read_orders(tmp_path, spark):
    df = pd.DataFrame({
        "Order ID": ["O1"],
        "Customer ID": ["C1"],
        "Product ID": ["P1"],
        "Profit": [10],
        "Order Date": ["1/1/2023"],
        "Ship Date": ["2/1/2023"]
    })
    path = tmp_path / "orders.json"
    df.to_json(path, orient="records", lines=False)

    spark_df = read_orders(spark, str(path))
    assert spark_df.count() == 1


# ======================================================
#        TEST CLEAN CUSTOMERS
# ======================================================

def test_clean_customers(customers_raw):
    df_clean, df_issues = clean_customers(customers_raw)

    # Duplicate ID removed
    assert df_clean.filter(col("Customer ID") == "C1").count() == 1

    # Missing ID removed
    assert df_clean.filter(col("Customer ID").isNull()).count() == 0

    # Missing name imputed
    assert df_clean.filter(col("Customer Name").startswith("IMPUTED_")).count() > 0

    # Validation errors recorded
    issues = df_issues.select("Issues").rdd.flatMap(lambda r: r["Issues"]).collect()
    assert "invalid_name" in issues
    assert "invalid_email" in issues
    assert "invalid_phone" in issues
    assert "invalid_postal" in issues
    assert "invalid_region" in issues
    assert "missing_name" in issues


# ======================================================
#        TEST CLEAN PRODUCTS
# ======================================================

def test_clean_products(products_raw):
    df_clean, df_issues = clean_products(products_raw)

    # Duplicate removed
    assert df_clean.filter(col("Product ID") == "P1").count() == 1

    # Missing ID removed
    assert df_clean.filter(col("Product ID").isNull()).count() == 0

    # Negative price issues
    assert df_issues.count() == 1
    assert "negative_unit_price" in df_issues.first()["Issues"]


# ======================================================
#        TEST CLEAN ORDERS
# ======================================================

def test_clean_orders(orders_raw, customers_raw, products_raw):
    df_customers_clean, _ = clean_customers(customers_raw)
    df_products_clean, _ = clean_products(products_raw)

    df_clean, df_issues = clean_orders(orders_raw, df_customers_clean, df_products_clean)

    # Duplicates removed
    assert df_clean.filter(col("Order ID") == "O1").count() == 1

    # Missing order ID removed
    assert df_clean.filter(col("Order ID").isNull()).count() == 0

    # Validation for invalid IDs
    issues = df_issues.select("Issues").rdd.flatMap(lambda r: r["Issues"]).collect()
    assert "invalid_customer_id" in issues
    assert "invalid_product_id" in issues


# ======================================================
#          TEST DIMENSION TABLES
# ======================================================

def test_build_customer_dim(customers_raw):
    df_clean, _ = clean_customers(customers_raw)
    df_dim = build_customer_dim(df_clean)

    assert "CustomerID" in df_dim.columns
    assert "CustomerName" in df_dim.columns
    assert "Country" in df_dim.columns


def test_build_product_dim(products_raw):
    df_clean, _ = clean_products(products_raw)
    df_dim = build_product_dim(df_clean)

    assert "ProductID" in df_dim.columns
    assert "Category" in df_dim.columns
    assert "SubCategory" in df_dim.columns


# ======================================================
#           TEST ENRICHED ORDERS
# ======================================================

def test_build_orders_enriched(orders_raw, customers_raw, products_raw):
    df_customers_clean, _ = clean_customers(customers_raw)
    df_products_clean, _ = clean_products(products_raw)
    df_clean_orders, _ = clean_orders(orders_raw, df_customers_clean, df_products_clean)

    df_cust_dim = build_customer_dim(df_customers_clean)
    df_prod_dim = build_product_dim(df_products_clean)

    df_enriched = build_orders_enriched(df_clean_orders, df_cust_dim, df_prod_dim)

    assert "CustomerName" in df_enriched.columns
    assert "Category" in df_enriched.columns
    assert "ProfitRounded" in df_enriched.columns
    assert "Year" in df_enriched.columns


# ======================================================
#           TEST AGGREGATION
# ======================================================

def test_aggregate_profit(orders_raw, customers_raw, products_raw):
    df_customers_clean, _ = clean_customers(customers_raw)
    df_products_clean, _ = clean_products(products_raw)
    df_clean_orders, _ = clean_orders(orders_raw, df_customers_clean, df_products_clean)

    df_cust_dim = build_customer_dim(df_customers_clean)
    df_prod_dim = build_product_dim(df_products_clean)
    df_enriched = build_orders_enriched(df_clean_orders, df_cust_dim, df_prod_dim)

    df_agg = aggregate_profit_by(df_enriched)

    assert df_agg.count() > 0
    assert "TotalProfit" in df_agg.columns
    assert df_agg.select("TotalProfit").first()[0] is not None

# COMMAND ----------

# DBTITLE 1,Aggregate and Enrichment Check
# ======================================================
#                SAMPLE DATA FIXTURES
# ======================================================

@pytest.fixture
def sample_customers(spark):
    return spark.createDataFrame([
        Row(**{"Customer ID": "C1", "Customer Name": "Alice", "email": "a@x.com",
               "phone": "1234567890", "Postal Code": "12345", "Region": "North", "Country": "USA"}),
        Row(**{"Customer ID": "C2", "Customer Name": "Bob", "email": "b@x.com",
               "phone": "1234567890", "Postal Code": "12345", "Region": "South", "Country": "USA"}),
    ])

@pytest.fixture
def sample_products(spark):
    return spark.createDataFrame([
        Row(**{"Product ID": "P1", "Category": "Office", "Sub-Category": "Paper",  "UnitPrice": float(2.50)}),
        Row(**{"Product ID": "P2", "Category": "Tech",   "Sub-Category": "Laptop","UnitPrice": float(800.0)}),
    ])

@pytest.fixture
def sample_orders(spark):
    return spark.createDataFrame([
        # Alice buys Paper twice
        Row(**{"Order ID": "O1", "Customer ID": "C1", "Product ID": "P1",
               "Profit": float(5.123), "Order Date": "1/1/2023", "Ship Date": "2/1/2023"}),
        Row(**{"Order ID": "O2", "Customer ID": "C1", "Product ID": "P1",
               "Profit": float(10.555), "Order Date": "3/1/2023", "Ship Date": "4/1/2023"}),

        # Bob buys a Laptop
        Row(**{"Order ID": "O3", "Customer ID": "C2", "Product ID": "P2",
               "Profit": float(100.987), "Order Date": "5/1/2023", "Ship Date": "6/1/2023"}),
    ])

# ======================================================
#           TEST ENRICHED ORDERS LAYER
# ======================================================

def test_enriched_layer(sample_orders, sample_customers, sample_products):

    df_cust_clean, _ = clean_customers(sample_customers)
    df_prod_clean, _ = clean_products(sample_products)
    df_orders_clean, _ = clean_orders(sample_orders, df_cust_clean, df_prod_clean)

    df_cust_dim = build_customer_dim(df_cust_clean)
    df_prod_dim = build_product_dim(df_prod_clean)

    df_enriched = build_orders_enriched(df_orders_clean, df_cust_dim, df_prod_dim)

    # ---------- Schema validation ----------
    expected_cols = {
        "Order ID", "OrderDate", "ShipDate",
        "CustomerID", "CustomerName", "Country",
        "ProductID", "Category", "SubCategory",
        "ProfitRounded", "Year"
    }
    assert expected_cols.issubset(set(df_enriched.columns))

    # ---------- Profit rounding validation ----------
    profits = df_enriched.select("ProfitRounded").rdd.flatMap(lambda r: r).collect()
    assert profits == [5.12, 10.56, 100.99]   # strict rounding

    # ---------- Join validation ----------
    row_o1 = df_enriched.filter(col("Order ID") == "O1").first()
    assert row_o1.CustomerName == "Alice"
    assert row_o1.Category == "Office"
    assert row_o1.SubCategory == "Paper"
    assert row_o1.Country == "USA"

# ======================================================
#           TEST AGGREGATE PROFIT LAYER
# ======================================================

def test_aggregate_profit_amount(sample_orders, sample_customers, sample_products):

    df_cust_clean, _ = clean_customers(sample_customers)
    df_prod_clean, _ = clean_products(sample_products)
    df_orders_clean, _ = clean_orders(sample_orders, df_cust_clean, df_prod_clean)

    df_cust_dim = build_customer_dim(df_cust_clean)
    df_prod_dim = build_product_dim(df_prod_clean)
    df_enriched = build_orders_enriched(df_orders_clean, df_cust_dim, df_prod_dim)

    df_agg = aggregate_profit_by(df_enriched)

    # ---------- Schema ----------
    expected_cols = {"Year", "Category", "SubCategory", "CustomerName", "TotalProfit"}
    assert expected_cols.issubset(set(df_agg.columns))

    # ---------- Values ----------
    results = {(r.Year, r.Category, r.SubCategory, r.CustomerName): r.TotalProfit
               for r in df_agg.collect()}

    # Alice (Paper)
    assert results[(2023, "Office", "Paper", "Alice")] == 15.68
    # Bob (Laptop)
    assert results[(2023, "Tech", "Laptop", "Bob")] == 100.99

    # Ensure only 2 rows in aggregate
    assert len(results) == 2

# COMMAND ----------

# DBTITLE 1,SQL related TCs
# ======================================================
#                SAMPLE DATA FIXTURE
# ======================================================

@pytest.fixture
def sample_enriched_data(spark):
    df = spark.createDataFrame([
        # Year 2023
        Row(Year=2023, CustomerName="Alice", Category="Office", TotalProfit=10.12),
        Row(Year=2023, CustomerName="Alice", Category="Office", TotalProfit=5.99),
        Row(Year=2023, CustomerName="Bob", Category="Tech", TotalProfit=50.56),
        # Year 2024
        Row(Year=2024, CustomerName="Alice", Category="Office", TotalProfit=7.33),
        Row(Year=2024, CustomerName="Charlie", Category="Tech", TotalProfit=15.78)
    ])
    df.createOrReplaceTempView("aggregates")
    return df

# ======================================================
#                  TEST CASES (SQL)
# ======================================================

def test_profit_by_year_sql(spark, sample_enriched_data):
    result = spark.sql("""
        SELECT Year, ROUND(SUM(TotalProfit), 2) AS ProfitByYear
        FROM aggregates
        GROUP BY Year
        ORDER BY Year
    """).collect()

    expected = {2023: 66.67, 2024: 23.11}  # 10.12+5.99+50.56=66.67
    for row in result:
        assert row['ProfitByYear'] == expected[row['Year']]


def test_profit_by_year_category_sql(spark, sample_enriched_data):
    result = spark.sql("""
        SELECT Year, Category, ROUND(SUM(TotalProfit), 2) AS ProfitByYearCategory
        FROM aggregates
        GROUP BY Year, Category
        ORDER BY Year, Category
    """).collect()

    expected = {(2023, "Office"): 16.11, (2023, "Tech"): 50.56,
                (2024, "Office"): 7.33, (2024, "Tech"): 15.78}

    for row in result:
        key = (row['Year'], row['Category'])
        assert row['ProfitByYearCategory'] == expected[key]


def test_profit_by_customer_sql(spark, sample_enriched_data):
    result = spark.sql("""
        SELECT CustomerName, ROUND(SUM(TotalProfit), 2) AS ProfitByCustomer
        FROM aggregates
        GROUP BY CustomerName
        ORDER BY ProfitByCustomer DESC
    """).collect()

    expected = {"Alice": 16.11+7.33, "Bob": 50.56, "Charlie": 15.78}
    expected = {k: round(v,2) for k,v in expected.items()}  # ensure rounded to 2 decimals

    for row in result:
        assert row['ProfitByCustomer'] == expected[row['CustomerName']]


def test_profit_by_customer_year_sql(spark, sample_enriched_data):
    result = spark.sql("""
        SELECT CustomerName, Year, ROUND(SUM(TotalProfit), 2) AS ProfitByCustomerYear
        FROM aggregates
        GROUP BY CustomerName, Year
        ORDER BY Year, CustomerName DESC
    """).collect()

    expected = {("Alice", 2023): 16.11, ("Alice", 2024): 7.33,
                ("Bob", 2023): 50.56, ("Charlie", 2024): 15.78}

    for row in result:
        key = (row['CustomerName'], row['Year'])
        assert row['ProfitByCustomerYear'] == expected[key]