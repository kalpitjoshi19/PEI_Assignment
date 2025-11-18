# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, regexp_replace, when, lit, array, explode, trim, upper,
    to_date, round as spark_round, year, array_remove, concat_ws, size, expr
)
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger("etl")
logger.setLevel(logging.INFO)

# ======================================================
#                    READERS
# ======================================================

def read_customers(spark: SparkSession, path: str) -> DataFrame:
    """
    Reads customer data from Excel (or CSV) using pandas
    and converts it into a Spark DataFrame.
    """
    pdf = pd.read_excel(path, engine="openpyxl")
    pdf["phone"] = pdf["phone"].astype(str)   # ensure phone column is string
    return spark.createDataFrame(pdf)


def read_products(spark: SparkSession, path: str) -> DataFrame:
    """Reads product CSV into Spark DataFrame."""
    pdf = pd.read_csv(path)
    return spark.createDataFrame(pdf)


def read_orders(spark: SparkSession, path: str) -> DataFrame:
    """Reads orders JSON into Spark DataFrame."""
    pdf = pd.read_json(path)
    return spark.createDataFrame(pdf)


# ======================================================
#                 CLEANING LOGIC
# ======================================================

def clean_customers(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Cleans the customer dataset and generates a data-quality issues table.

    - Removes null/duplicate Customer ID/null customer name
    - Standardizes trimming/casting
    - Validates:
        * Customer Name (Unicode letters + space)
        * Email format
        * Phone format (4 strict patterns)
        * US postal code (ZIP / ZIP+4)
        * Region (North/East/South/West/Central)
    - Issues column contains array of issue codes (no nulls)
    """

    # --- Drop invalid ID rows ---
    df = df.dropna(subset=["Customer ID"]).dropDuplicates(["Customer ID"])

    # --- Normalize string fields ---
    string_cols = ["Customer Name", "email", "phone", "Postal Code", "Region"]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c).cast("string")))

    # --- Build validation issues array ---
    df = df.withColumn(
        "Issues",
        array(
            when(col("Customer Name").isNull() | (col("Customer Name") == ""), lit("missing_name")),
            when(~col("Customer Name").rlike(r"^[\p{L} ]+$"), lit("invalid_name")),
            when(~col("email").rlike(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"), lit("invalid_email")),
            when(
                ~col("phone").rlike(
                    r"^(\d{10}|\(\d{3}\)-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}|\d{3}-\d{3}-\d{4})$"
                ),
                lit("invalid_phone")
            ),
            when(~col("Postal Code").rlike(r"^\d{5}(-\d{4})?$"), lit("invalid_postal")),
            when(~upper(col("Region")).isin(["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]), lit("invalid_region"))
        )
    )

    # --- Remove null entries from Issues array ---
    df = df.withColumn("Issues", expr("filter(Issues, x -> x IS NOT NULL)"))

    # Uncomment for debugging
    # display(df)

    # --- Build issue report subset ---
    issue_report = (
        df.select(
            col("Customer ID").alias("CustomerID"),
            col("Customer Name").alias("CustomerName"),
            col("email").alias("Email"),
            col("phone").alias("Phone"),
            col("Postal Code").alias("PostalCode"),
            col("Region"),
            col("Issues")
        )
        .where(size(col("Issues")) > 0)
    )

    #print("BEFORE - Records for Customer ID LC-16870")
    #display(df.filter(col("Customer ID") == "LC-16870"))

    # --- Consider only rows with Customer Name present ---
    # df = df.filter(col("Customer Name").isNotNull() & (trim(col("Customer Name")) != ""))

    # impute missing value for "Customer Name" from email column 
    df = df.withColumn(
        "Customer Name",
        when(
            col("Customer Name").isNull() | (trim(col("Customer Name")) == ""),
            concat_ws(
                "",
                lit("IMPUTED_"),
                regexp_extract(col("email"), r"^([^@]+)", 1)
            )
        ).otherwise(col("Customer Name"))
    )

    #print("AFTER - Records for Customer ID LC-16870")
    #display(df.filter(col("Customer ID") == "LC-16870"))

    return df, issue_report


def clean_products(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Cleans product data and returns (clean_df, issues_df)."""

    df = df.dropna(subset=["Product ID"]).dropDuplicates(["Product ID"])

    # Normalize price column name
    for c in df.columns:
        if c.lower().strip() in ("price per product", "price", "unitprice", "unit_price"):
            df = df.withColumnRenamed(c, "UnitPrice")
            break

    df = df.withColumn("UnitPrice", col("UnitPrice").cast("double"))

    # Identify invalid prices
    df_issues = df.filter(col("UnitPrice") < 0).withColumn(
        "Issues", array(lit("negative_unit_price"))
    )

    df_clean = df.filter(col("UnitPrice") >= 0)
    return df_clean, df_issues


def clean_orders(df_orders: DataFrame, df_customers: DataFrame, df_products: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Clean orders dataset, detect invalid Product/Customer IDs, and generate issue report."""

    # Drop rows with nulls in required columns
    df = df_orders.dropna(
        subset=["Order ID", "Customer ID", "Product ID", "Profit", "Order Date"]
    )

    # Convert dates
    df = df.withColumn("OrderDate", to_date(col("Order Date").cast("string"), "d/M/yyyy"))
    df = df.withColumn("ShipDate", to_date(col("Ship Date").cast("string"), "d/M/yyyy"))

    # Get valid IDs from input DataFrames
    valid_customer_ids = [row["Customer ID"] for row in df_customers.select("Customer ID").distinct().collect()]
    valid_product_ids = [row["Product ID"] for row in df_products.select("Product ID").distinct().collect()]

    # Add Issues array
    df = df.withColumn(
        "Issues",
        array(
            when(~col("Customer ID").isin(valid_customer_ids), lit("invalid_customer_id")),
            when(~col("Product ID").isin(valid_product_ids), lit("invalid_product_id"))
        )
    )
    df = df.withColumn("Issues", expr("filter(Issues, x -> x IS NOT NULL)"))

    # Issue report: only rows with issues
    issue_report = (
        df.select(
            col("Order ID").alias("OrderID"),
            col("Customer ID").alias("CustomerID"),
            col("Product ID").alias("ProductID"),
            col("OrderDate"),
            col("ShipDate"),
            col("Issues")
        )
        .where(size(col("Issues")) > 0)
    )

    # Only keep valid orders
    df_clean = df.where(size(col("Issues")) == 0)

    return df_clean, issue_report


# ======================================================
#         DIMENSION TABLES & ENRICHMENT
# ======================================================

def build_customer_dim(df_customers: DataFrame) -> DataFrame:
    """Builds Customer dimension table."""

    for c in df_customers.columns:
        cc = c.strip().lower()
        if cc in ("customer id", "customer_id"):
            id_col = c
        if cc in ("customer name", "name", "customer_name"):
            name_col = c
        if cc in ("country", "country_name"):
            country_col = c

    return df_customers.select(
        col(id_col).alias("CustomerID"),
        col(name_col).alias("CustomerName"),
        col(country_col).alias("Country")
    )


def build_product_dim(df_products: DataFrame) -> DataFrame:
    """Builds Product dimension table."""
    return df_products.select(
        col("Product ID").alias("ProductID"),
        col("Category"),
        col("Sub-Category").alias("SubCategory")
    )


def build_orders_enriched(df_orders, df_cust_dim, df_prod_dim):
    """Joins orders with customer & product dimensions."""

    df = df_orders.withColumn(
        "ProfitRounded", spark_round(col("Profit").cast("double"), 2)
    )

    df = df.withColumnRenamed("Customer ID", "CustomerID") \
           .withColumnRenamed("Product ID", "ProductID")

    if "OrderDate" not in df.columns:
        df = df.withColumn(
            "OrderDate", to_date(col("Order Date").cast("string"), "d/M/yyyy")
        )

    df_joined = (
        df.join(df_cust_dim, "CustomerID", "left")
          .join(df_prod_dim, "ProductID", "left")
    )

    df_joined = df_joined.withColumn("Year", year(col("OrderDate")))

    select_cols = [
        "Order ID", "OrderDate", "ShipDate", "Ship Mode",
        "CustomerID", "CustomerName", "Country",
        "ProductID", "Category", "SubCategory",
        "Quantity", "Price", "Discount",
        "ProfitRounded", "Year"
    ]

    existing_cols = [c for c in select_cols if c in df_joined.columns]
    return df_joined.select(*existing_cols)


def aggregate_profit_by(df_enriched):
    """Aggregates total profit by Year, Category, SubCategory, CustomerName."""
    return (
        df_enriched.groupBy("Year", "Category", "SubCategory", "CustomerName")
            .sum("ProfitRounded")
            .withColumnRenamed("sum(ProfitRounded)", "TotalProfit")
            .withColumn("TotalProfit", spark_round(col("TotalProfit"), 2))
    )


# ======================================================
#                     WRITERS
# ======================================================

def write_parquet(df, path: str, mode: str = "overwrite"):
    """Writes a DataFrame to Parquet."""
    df.write.mode(mode).parquet(path)


# ======================================================
#                     MAIN ETL
# ======================================================

def run_etl(
    spark, customers_path, products_path, orders_path, output_base_location
):
    logger.info("Reading raw files...")
    df_customers = read_customers(spark, customers_path)
    df_products = read_products(spark, products_path)
    df_orders = read_orders(spark, orders_path)

    print("BEFORE - count df_customers")
    display(df_customers.count())

    logger.info("Cleaning...")
    df_customers_clean, df_customer_issues = clean_customers(df_customers)
    df_products_clean, df_product_issues = clean_products(df_products)
    df_orders_clean, df_order_issues = clean_orders(
        df_orders, df_customers_clean, df_products_clean
    )

    print("AFTER - clean_customers")
    display(df_customers_clean.count())

    logger.info("Building dimensions...")
    df_customer_dim = build_customer_dim(df_customers_clean)
    df_product_dim = build_product_dim(df_products_clean)

    logger.info("Enriching orders...")
    df_orders_enriched = build_orders_enriched(
        df_orders_clean, df_customer_dim, df_product_dim
    )

    print("AFTER - build_orders_enriched")
    display(df_orders_enriched)

    logger.info("Writing cleaned outputs...")
    write_parquet(df_customer_issues, f"{output_base_location}/customer_data_issues")
    write_parquet(df_product_issues, f"{output_base_location}/product_data_issues")
    write_parquet(df_order_issues, f"{output_base_location}/order_data_issues")
    write_parquet(df_customer_dim, f"{output_base_location}/customers")
    write_parquet(df_product_dim, f"{output_base_location}/products")
    write_parquet(df_orders_enriched, f"{output_base_location}/orders_enriched")

    logger.info("Aggregating...")
    df_agg = aggregate_profit_by(df_orders_enriched)
    write_parquet(df_agg, f"{output_base_location}/aggregates")

    return {
        "customers": df_customer_dim,
        "products": df_product_dim,
        "orders_enriched": df_orders_enriched,
        "customer_data_issues": df_customer_issues,
        "product_data_issues": df_product_issues,
        "order_data_issues": df_order_issues,
        "aggregates": df_agg
    }

# COMMAND ----------

customers_file = "/Workspace/Users/kalpitjoshi19@gmail.com/PEI_test/input_files/Customer.xlsx"
products_file = "/Workspace/Users/kalpitjoshi19@gmail.com/PEI_test/input_files/Products.csv"
orders_file = "/Workspace/Users/kalpitjoshi19@gmail.com/PEI_test/input_files/Orders.json"
output_base_location = "/Volumes/pei/default/output"

# COMMAND ----------

run_etl(spark, customers_file, products_file, orders_file, output_base_location)

# COMMAND ----------

# List files in output directory
dbutils.fs.ls("/Volumes/pei/default/output")

# Data Issue Tables
df_customer_data_issues = spark.read.parquet("/Volumes/pei/default/output/customer_data_issues")
displayHTML(f"<h2>Customer Data Issues (Sample) - Total: {df_customer_data_issues.count()}</h2>")
display(df_customer_data_issues.limit(5))

df_product_data_issues = spark.read.parquet("/Volumes/pei/default/output/product_data_issues")
displayHTML(f"<h2>Product Data Issues (Sample) - Total: {df_product_data_issues.count()}</h2>")
display(df_product_data_issues.limit(5))

df_order_data_issues = spark.read.parquet("/Volumes/pei/default/output/order_data_issues")
displayHTML(f"<h2>Order Data Issues (Sample) - Total: {df_order_data_issues.count()}</h2>")
display(df_order_data_issues.limit(5))

# Dimension Tables
df_customers_dim = spark.read.parquet("/Volumes/pei/default/output/customers")
displayHTML(f"<h2>Customer Dimension (Sample) - Total: {df_customers_dim.count()}</h2>")
display(df_customers_dim.limit(5))

df_products_dim = spark.read.parquet("/Volumes/pei/default/output/products")
displayHTML(f"<h2>Product Dimension (Sample) - Total: {df_products_dim.count()}</h2>")
display(df_products_dim.limit(5))

# Fact Table
df_orders_enriched = spark.read.parquet("/Volumes/pei/default/output/orders_enriched")
displayHTML(f"<h2>Orders Enriched (Sample) - Total: {df_orders_enriched.count()}</h2>")
display(df_orders_enriched.limit(5))

# Aggregate Table
df_agg = spark.read.parquet("/Volumes/pei/default/output/aggregates")
displayHTML(f"<h2>Aggregates (Sample) - Total: {df_agg.count()}</h2>")
display(df_agg.limit(5))

# COMMAND ----------

# Register the aggregate DataFrame as a SQL view
df_agg.createOrReplaceTempView("aggregates")

# Profit by Year
displayHTML("<h2 style='color:navy;font-size:24px;'>Profit by Year (Sample)</h2>")
display(
    spark.sql("""
        SELECT Year, ROUND(SUM(TotalProfit), 2) AS ProfitByYear
        FROM aggregates
        GROUP BY Year
        ORDER BY Year
    """).limit(5)
)

# Profit by Year + Product Category
displayHTML("<h2 style='color:navy;font-size:24px;'>Profit by Year and Product Category (Sample)</h2>")
display(
    spark.sql("""
        SELECT Year, Category, ROUND(SUM(TotalProfit), 2) AS ProfitByYearCategory
        FROM aggregates
        GROUP BY Year, Category
        ORDER BY Year, Category
    """).limit(5)
)

# Profit by Customer
displayHTML("<h2 style='color:navy;font-size:24px;'>Profit by Customer (Sample)</h2>")
display(
    spark.sql("""
        SELECT CustomerName, ROUND(SUM(TotalProfit), 2) AS ProfitByCustomer
        FROM aggregates
        GROUP BY CustomerName
        ORDER BY ProfitByCustomer DESC
    """).limit(5)
)

# Profit by Customer + Year
displayHTML("<h2 style='color:navy;font-size:24px;'>Profit by Customer and Year (Sample)</h2>")
display(
    spark.sql("""
        SELECT CustomerName, Year, ROUND(SUM(TotalProfit), 2) AS ProfitByCustomerYear
        FROM aggregates
        GROUP BY CustomerName, Year
        ORDER BY Year, CustomerName DESC
    """).limit(5)
)