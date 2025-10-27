# Pyspark-power-predictor
# ⚡ Household Power Consumption Predictor

This is a Streamlit web application that uses **Apache PySpark** to perform time-series analysis and predict household power consumption. The app loads a large dataset, performs feature engineering, trains a **Random Forest Regressor** model using PySpark's MLlib, and visualizes the actual vs. predicted results in an interactive chart.



---

## 🚀 Features

* **File Upload:** Allows users to upload their own power consumption CSV or TXT files.
* **Separator Selection:** Lets the user choose the correct delimiter (`,` or `;`), which is critical for the recommended dataset.
* **Big Data Processing:** Uses PySpark to efficiently load, preprocess, and feature-engineer data that might be too large for in-memory tools like Pandas.
* **Time-Series Feature Engineering:**
    * **Lag Features:** Creates `Prev_Global_active_power` and `Prev_Global_intensity`.
    * **Moving Averages:** Calculates 10-minute and 60-minute moving averages (`MA_10min`, `MA_60min`).
* **Machine Learning:** Trains a `RandomForestRegressor` model on the engineered features to predict `Global_active_power`.
* **Model Evaluation:** Reports model performance using **RMSE** (Root Mean Squared Error) and **R²** (R-squared).
* **Interactive Visualization:** Uses Plotly to display an interactive, zoomable line chart comparing actual vs. predicted power consumption.

---

## 🛠️ Tech Stack

* **Core:** Python 3.8+
* **Web Framework:** Streamlit
* **Data Processing/ML:** Apache PySpark (specifically `pyspark.sql` and `pyspark.ml`)
* **Data Handling:** Pandas
* **Plotting:** Plotly (via `plotly.express`)

---

## 🏁 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

**This project will not run without Java.** PySpark is built on top of the Java Virtual Machine (JVM).

* **Java Development Kit (JDK):** You must install a JDK, version 8 or 11.
    * We recommend the [Microsoft Build of OpenJDK 11](https://www.microsoft.com/openjdk).
* **`JAVA_HOME` Environment Variable:** After installing the JDK, you **must** set the `JAVA_HOME` environment variable on your system.
    * **Windows:** Set `JAVA_HOME` to the path of your JDK, e.g., `C:\Program Files\Microsoft\jdk-11.x.x.x`
    * **macOS/Linux:** Set `JAVA_HOME` in your `.bashrc` or `.zshrc` file, e.g., `export JAVA_HOME=/usr/lib/jvm/ms-jdk-11-linux-x64`

### 2. Project Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mr-TarunD/Pyspark-power-predictor.git
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it
    # Windows (PowerShell)
    .\venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Create a `requirements.txt` file:**
    Create a file named `requirements.txt` in your project's root directory and add the following libraries:
    ```
    streamlit
    pyspark
    pandas
    plotly
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Get the Data

This application is designed to work with the **"Individual household electric power consumption"** dataset from the UCI Machine Learning Repository.

1.  **Download the data** from [this link](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2.  Unzip the file. You will get a file named `household_power_consumption.txt`. This is the file you will upload to the app.
3.  **Note:** This dataset uses a **semicolon (`;`)** as its separator.

### 4. Run the Application

In your terminal (with the virtual environment activated), run:

```bash
streamlit run app.py
