# Building a Recipe Recommender System with VAE

### Introduction 

The reviews dataset contains **1,401,982 reviews** from **271,907 users**, providing details like author, rating, review text, and more.

### Workflow Overview

The steps for building the recommender system are as follows:

1. **Data Download and Cleaning**  
   We first download the datasets and clean them, handling missing values and inconsistencies.


2. **Preprocessing**  
   - Scale numeric data  
   - Create vector embeddings for textual data such as ingredients and instructions


3. **Modeling with Variational Autoencoder (VAE)**  
   - Build a VAE to learn latent representations of recipes  
   - Optimize hyperparameters for better performance  
   - Create a latent space for all recipes


4. **Recommendation**  
   Based on liked recipes, recommend similar recipes using the learned latent space.


We also provide a notebook analysis explaining **why machine learning is useful for this task**, and why traditional non-ML methods may be weaker for large and complex datasets. 

### Part 1: Data Exploration and Preprocessing

All the data exploration and preprocessing analysis can be found in the following notebook:  

📁 File: `Data_Exploration_Preparation.ipynb`  
🔗 Source: [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/Data_Exploration_Preparation.ipynb)

After downloading the data from Kaggle (`irkaal/foodcom-recipes-and-reviews`), we load them into CSV files. The **recipe dataframe** contains the following columns:

- `RecipeId`, `Name`, `AuthorId`, `AuthorName`, `CookTime`, `PrepTime`, `DatePublished`, `Description`, `Images`, `RecipeCategory`, `Keywords`  
- `RecipeIngredientQuantities`, `ReviewCount`, `Calories`, `FatContent`, `SaturatedFatContent`, `CholesterolContent`, `SodiumContent`, `CarbohydrateContent`, `FiberContent`, `SugarContent`, `ProteinContent`  
- `RecipeServings`, `RecipeYield`, `RecipeInstructions`

> In this project, we focus only on the recipes; we do not use the reviews dataset.

The first step in preprocessing is **exploring missing values**. We inspect the dataset to find columns with many `NaN` values and remove them to simplify further analysis. 

#### Selecting Important Features and Handling Outliers

After removing columns with too many missing values, we focus on the most relevant features for our model.

**Numeric columns we keep**:

- `Calories`
- `FatContent`
- `SaturatedFatContent`
- `CholesterolContent`
- `SodiumContent`
- `CarbohydrateContent`
- `FiberContent`
- `SugarContent`
- `ProteinContent`

**Text columns we keep**:

- `Name`
- `RecipeCategory`
- `RecipeIngredientParts`
- `RecipeInstructions`

Some numeric columns contain **outliers**. To handle them, we retain only the data points that fall within the **99.5th percentile** for each column. This helps prevent extreme values from skewing the model training while keeping almost all of the data.
