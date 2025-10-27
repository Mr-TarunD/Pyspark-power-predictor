import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, avg, when, to_timestamp, concat_ws
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------
# Configure Spark temp dir (Cross-platform)
# -----------------------------
# Replaced hardcoded C:\ path with a cross-platform temporary directory
temp_dir = os.path.join(tempfile.gettempdir(), "spark-temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ['SPARK_LOCAL_DIRS'] = temp_dir

# -----------------------------
# Initialize Spark (with Streamlit Caching)
# -----------------------------
# We cache the Spark session so it doesn't restart on every widget change
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("PowerConsumptionPrediction") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

spark = init_spark()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⚡ Household Power Consumption Predictor")
st.write("This app uses PySpark and a Random Forest Regressor to predict `Global_active_power`.")

# Added a separator choice because power data often uses ';'
sep = st.selectbox("Select the CSV Separator", (';', ','))

uploaded_file = st.file_uploader("Upload your power consumption CSV file", type=["csv", "txt"])

if uploaded_file:
    # We must save the uploaded file to a temporary path
    # so that PySpark can read it from the filesystem.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_csv_path = tmp_file.name

    try:
        with st.spinner("Loading and Processing Data with Spark..."):
            # -----------------------------
            # Load CSV in PySpark
            # -----------------------------
            df = spark.read.csv(temp_csv_path,
                                header=True,
                                sep=sep,
                                nullValue='?', # Handle missing values
                                inferSchema=False) # We'll cast types manually

            st.write("Preview of loaded data (top 5 rows):", df.limit(5).toPandas())

            # -----------------------------
            # Preprocessing (Adapted for Power Data)
            # -----------------------------
            
            # 1. Create a single timestamp
            df = df.withColumn("timestamp_str", concat_ws(" ", col("Date"), col("Time")))
            # Assuming d/M/y format as in the original dataset
            df = df.withColumn("timestamp", to_timestamp(col("timestamp_str"), "d/M/y H:m:s"))
            
            # 2. Cast all relevant columns to double
            numeric_cols = [
                'Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
            ]
            for c in numeric_cols:
                df = df.withColumn(c, col(c).cast('double'))

            # 3. Drop rows with nulls (especially our target and timestamp)
            df_clean = df.na.drop(subset=["timestamp", "Global_active_power"])

            # -----------------------------
            # Feature Engineering (Adapted from your stock logic)
            # -----------------------------
            
            # The window is now ordered by time, with NO partition (it's one series)
            w = Window.orderBy('timestamp')
            
            # Create lag features (like your 'Prev_Close' and 'Prev_Volume')
            df_features = df_clean.withColumn('Prev_Global_active_power', lag('Global_active_power', 1).over(w))
            df_features = df_features.withColumn('Prev_Global_intensity', lag('Global_intensity', 1).over(w))

            # Create Moving Average features (like your 'MA_5', 'MA_10')
            # We'll use 10-minute and 60-minute windows
            df_features = df_features.withColumn('MA_10min', avg('Global_active_power').over(w.rowsBetween(-9, 0)))
            df_features = df_features.withColumn('MA_60min', avg('Global_active_power').over(w.rowsBetween(-59, 0)))

            # Drop nulls created by the lag/window functions
            df_features = df_features.na.drop()

            # -----------------------------
            # Assemble features
            # -----------------------------
            # These are the new features for the power model
            feature_cols = [
                'Prev_Global_active_power', 'Prev_Global_intensity',
                'MA_10min', 'MA_60min',
                'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
            ]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid="skip")
            df_model_data = assembler.transform(df_features)

            # -----------------------------
            # Train-Test Split (Using your randomSplit logic)
            # -----------------------------
            train_df, test_df = df_model_data.randomSplit([0.8, 0.2], seed=42)

            st.write(f"Data processed. Training on {train_df.count()} samples, testing on {test_df.count()} samples.")

        # -----------------------------
        # Train Random Forest
        # -----------------------------
        with st.spinner("Training Random Forest model... (This may take a minute)"):
            rf = RandomForestRegressor(
                featuresCol='features',
                labelCol='Global_active_power', # Our new target
                numTrees=100, # Reduced trees for faster app demo
                maxDepth=8    # Reduced depth for faster app demo
            )
            model = rf.fit(train_df)
            
        # -----------------------------
        # Predictions & Evaluation
        # -----------------------------
        with st.spinner("Evaluating model..."):
            predictions = model.transform(test_df)
            
            evaluator_rmse = RegressionEvaluator(labelCol='Global_active_power', predictionCol='prediction', metricName='rmse')
            evaluator_r2 = RegressionEvaluator(labelCol='Global_active_power', predictionCol='prediction', metricName='r2')
            
            rmse = evaluator_rmse.evaluate(predictions)
            r2 = evaluator_r2.evaluate(predictions)
            
            st.success(f"Model Performance: RMSE = {rmse:.4f} | R² = {r2:.4f}")

            # -----------------------------
            # Plot Ticker Actual vs Predicted
            # -----------------------------
            st.subheader("Actual vs. Predicted Power (Sample from Test Set)")
            st.write("Plotting a sorted sample of 5,000 test points for clarity.")

            # Sample the predictions to avoid crashing Streamlit with .toPandas()
            pred_pd_sample = predictions.select('timestamp', 'Global_active_power', 'prediction') \
                                        .orderBy('timestamp') \
                                        .limit(5000) \
                                        .toPandas()

            # Switched to Plotly for interactive charts (zooming/panning)
            fig = px.line(
                pred_pd_sample,
                x='timestamp',
                y=['Global_active_power', 'prediction'],
                title="Global Active Power: Actual vs. Predicted",
                labels={'value': 'Power (kilowatts)', 'variable': 'Legend'}
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check the console for more details. Ensure the separator is correct and the file format matches.")
        print(e) # Log the full error to the console

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)