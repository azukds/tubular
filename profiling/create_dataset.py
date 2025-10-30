import pandas as pd
from sklearn.datasets import fetch_california_housing


def create_standard_pandas_dataset() -> pd.DataFrame:
    """Function to create the california housing dataset and alter it so that it can be transformed on by every transformer in fubular."""
    # Load dataset
    cali = fetch_california_housing()
    cali_df = pd.DataFrame(cali["data"], columns=cali["feature_names"])

    # Sample to introduce missing vals
    cali_df["AveOccup"] = cali_df["AveOccup"].sample(frac=0.99, random_state=1)
    cali_df["HouseAge"] = cali_df["HouseAge"].sample(frac=0.95, random_state=2)
    cali_df["Population"] = cali_df["Population"].sample(frac=0.995, random_state=3)

    # Create a mock categorical feature same length as cali dataset
    cat_list = ["a", "b", "c", "d", "e", "f", "g", "d", "h", "a"] * 2064
    # Introduce rare level
    cat_list[20] = "z"
    # Fewer categorical leveles for OHE
    cat_list_ohe = ["a", "b"] * 2064 * 5

    # Creating copies of levels so that later transformers still have e.g. missing values to impute
    for i in ["1", "2", "3", "4"]:
        cali_df["categorical_" + i] = cat_list

    cali_df["categorical_ohe"] = cat_list_ohe

    for i in range(1, 8):
        str_i = str(i)
        cali_df[["HouseAge_" + str_i, "AveOccup_" + str_i, "Population_" + str_i]] = (
            cali_df[["HouseAge", "AveOccup", "Population"]].copy()
        )

    return cali_df
