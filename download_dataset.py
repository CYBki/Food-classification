"""Download sample pizza/steak/sushi dataset."""
from helper_functions import download_data

if __name__ == "__main__":
    download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )
