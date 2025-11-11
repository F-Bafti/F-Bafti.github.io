# Building a Recipe Recommender System with VAE
author: "Fahimeh Baftizadeh"
date: 2025-11-11


In this blog post, we will explore a technique to build a **recipe recommender system**.

The data was obtained from Kaggle, originally scraped from [food.com](https://www.food.com). It contains **522,517 recipes** from **312 different categories**. Each recipe includes information such as cooking times, servings, ingredients, nutrition, instructions, and more.

The reviews dataset contains **1,401,982 reviews** from **271,907 users**, providing details like author, rating, review text, and more.

## Data Cleaning and Preparation

Before building our model, we need to clean and prepare the data. All the steps of cleaning and preparation are explained in the notebook [`Data_exploration_Preparation.ipynb`](Data_exploration_Preparation.ipynb).

Example of setting up the Kaggle API and downloading the dataset:

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset_path = api.dataset_download_files("irkaal/foodcom-recipes-and-reviews", path="data", unzip=True)
