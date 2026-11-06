
import pandas as pd

DATA_PATH = "data/Section7_Q4_Ecommerce_Product_Return_CICD.csv"


def load_and_preprocess():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove Order ID
    df = df.drop(columns=["Order_ID"])

    # Handle invalid values
    df.loc[df["Product_Price_INR"] <= 0, "Product_Price_INR"] = None

    df.loc[
        (df["Discount_Percent"] < 0) |
        (df["Discount_Percent"] > 100),
        "Discount_Percent"
    ] = None

    df.loc[
        (df["Customer_Rating_PreOrder"] < 1) |
        (df["Customer_Rating_PreOrder"] > 5),
        "Customer_Rating_PreOrder"
    ] = None

    df.loc[df["Actual_Delivery_Days"] <= 0,
           "Actual_Delivery_Days"] = None

    df.loc[df["Expected_Delivery_Days"] <= 0,
           "Expected_Delivery_Days"] = None

    df.loc[
        (df["Previous_Returns"] < 0) |
        (df["Previous_Returns"] > df["Previous_Orders"]),
        "Previous_Returns"
    ] = None

    # Feature Engineering 1: Discount Amount
    df["Discount_Amount"] = (
        df["Product_Price_INR"] *
        df["Discount_Percent"] / 100
    )

    # Feature Engineering 2: Delivery Delay
    df["Delivery_Delay"] = (
        df["Actual_Delivery_Days"] -
        df["Expected_Delivery_Days"]
    )

    # Feature Engineering 3: Return Frequency
    df["Customer_Return_Frequency"] = (
        df["Previous_Returns"] /
        df["Previous_Orders"].replace(0, 1)
    )

    # Encode target variable
    df["Return_Status"] = df["Return_Status"].map({
        "Not Returned": 0,
        "Returned": 1
    })

    # Remove invalid target records
    df = df.dropna(subset=["Return_Status"])

    print("\nPreprocessing completed!")
    print("Processed dataset shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["Return_Status"].value_counts())

    print("\nProcessed dataset:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_and_preprocess()
