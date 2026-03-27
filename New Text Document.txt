import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.data.loader import BookCrossingLoader
from src.data.preprocessor import BookCrossingPreprocessor

# Step 1: Load raw data
loader = BookCrossingLoader("data/raw")
raw = loader.load_all()

# Step 2: Clean and preprocess
prep = BookCrossingPreprocessor()
clean = prep.fit_transform(raw)

# Step 3: Export libsvm files (used by all 3 tasks)
prep.export_libsvm(clean, output_dir="data/processed")

# Step 4: Export cleaned CSVs (for inspection and notebooks)
prep.export_cleaned_csvs(clean, output_dir="data/processed")

# Step 5: Print final summary
print(prep.summary(clean))