#test of prepare the data to the model
import pytest
import pandas as pd
from cox import prepare_data_to_model



def test_prepare_data_to_model():
    final_df = pd.read_csv("data_prossesing_final.csv").set_index('sample number')
    data = prepare_data_to_model(final_df)
    assert data.shape[1] > final_df.shape[1], "Number of columns should be more after preparation"

# Run pytest
if __name__ == "__main__":
    pytest.main()
