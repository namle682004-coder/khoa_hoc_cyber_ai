import numpy as np
import pandas as pd
import joblib
from sklearn.utils._array_api import get_namespace

try:
    model = joblib.load('outputs/models/best_model.pkl')
    preprocessor = model.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    
    print("Features:", cat_encoder.feature_names_in_)
    for i, cat in enumerate(cat_encoder.categories_):
        print(f"Category {i} ({cat_encoder.feature_names_in_[i]}): {cat.dtype} {cat}")

    df = pd.DataFrame({
        'Administrative': 0, 'Administrative_Duration': 0.0, 'Informational': 0, 'Informational_Duration': 0.0,
        'ProductRelated': 1, 'ProductRelated_Duration': 10.0, 'BounceRates': 0.0, 'ExitRates': 0.0,
        'PageValues': 0.0, 'SpecialDay': 0.0, 'Month': 'Feb', 'OperatingSystems': 1, 'Browser': 1,
        'Region': 1, 'TrafficType': 1, 'VisitorType': 'Returning_Visitor', 'Weekend': False
    })
    
    # Matching types in main.py
    df["Month"] = df["Month"].astype("object")
    df["VisitorType"] = df["VisitorType"].astype("object")
    df["Weekend"] = df["Weekend"].astype("bool")
    
    print("\nDataFrame dtypes:")
    print(df.dtypes)
    
    # Test Weekend specifically
    values = df['Weekend'].values
    xp, _ = get_namespace(values)
    print(f"\nWeekend values dtype: {values.dtype}")
    print(f"Is numeric (xp.isdtype): {xp.isdtype(values.dtype, 'numeric')}")
    
    # Select categories for Weekend
    weekend_cat = cat_encoder.categories_[2] # Assuming Weekend is the 3rd categorical feature
    print(f"Weekend categories ({weekend_cat.dtype}): {weekend_cat}")
    
    try:
        print("Attempting xp.isnan(weekend_cat)...")
        print(xp.isnan(weekend_cat))
    except Exception as e:
        print(f"xp.isnan(weekend_cat) failed: {e}")

except Exception as e:
    print(f"Error in debug script: {e}")
    import traceback
    traceback.print_exc()
