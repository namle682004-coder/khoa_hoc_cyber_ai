import joblib

try:
    model = joblib.load('outputs/models/best_model.pkl')
    preprocessor = model.named_steps['preprocessor']
    
    for name, transformer, cols in preprocessor.transformers_:
        print(f"Transformer '{name}':")
        print(f"Columns: {cols}")
        if hasattr(transformer, 'categories_'):
             print(f"Categories sample: {transformer.categories_[0][:5] if transformer.categories_ else 'None'}")
        print("-" * 20)

except Exception as e:
    print(f"Error: {e}")
