# 🧁 Recipe Recommender System with VAE

### Introduction  

The reviews dataset contains **1,401,982 reviews** from **271,907 users**, providing details like author, rating, review text, and more.  

### Workflow Overview  

The steps for building the recommender system are as follows:

1. **Data Download and Cleaning**  
   Download and clean the datasets, handling missing values and inconsistencies.

2. **Preprocessing**  
   - Scale numeric data  
   - Create vector embeddings for textual data (ingredients, instructions)  
   - Decide why to use a VAE-based model  

3. **Modeling with Variational Autoencoder (VAE)**  
   - Build a VAE to learn latent representations of recipes  
   - Optimize hyperparameters for better performance  
   - Create a latent space for all recipes  

4. **Recommendation**  
   Recommend similar recipes based on the learned latent space.  

We also provide a notebook explaining **why machine learning is essential** for this task, especially for handling large and complex datasets.  

---

### 🧩 Part 1: Data Exploration and Preprocessing  

📁 **File:** `Data_Exploration_Preparation.ipynb`  
🔗 [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/Data_Exploration_Preparation.ipynb)  

After downloading the data from Kaggle (`irkaal/foodcom-recipes-and-reviews`), we load them into CSV files. The **recipe dataframe** contains:  

`RecipeId`, `Name`, `AuthorId`, `AuthorName`, `CookTime`, `PrepTime`, `DatePublished`, `Description`, `Images`, `RecipeCategory`, `Keywords`, `RecipeIngredientQuantities`, `ReviewCount`, `Calories`, `FatContent`, `SaturatedFatContent`, `CholesterolContent`, `SodiumContent`, `CarbohydrateContent`, `FiberContent`, `SugarContent`, `ProteinContent`, `RecipeServings`, `RecipeYield`, `RecipeInstructions`.

> We focus only on the recipes dataset and do not use the reviews.  

We first explore missing values and remove columns with excessive `NaN`s to simplify analysis.  

---

### ⚙️ Part 2: Selecting Important Features and Handling Outliers  

We focus on the most relevant features for our model.  

**Numeric columns kept:**  
Calories, FatContent, SaturatedFatContent, CholesterolContent, SodiumContent, CarbohydrateContent, FiberContent, SugarContent, ProteinContent  

**Text columns kept:**  
Name, RecipeCategory, RecipeIngredientParts, RecipeInstructions  

Outliers are trimmed to the **99.5th percentile** for each numeric feature, keeping nearly all data while avoiding skew.  

---

### 🧠 2.1 Text Embeddings with `all-MiniLM-L6-v2`

To process unstructured text (recipe names, ingredients, and instructions), we use the **`all-MiniLM-L6-v2`** pre-trained model to generate 384-dimensional embeddings.  

**Why MiniLM?**  
- Compact and efficient (distilled model of BERT)  
- Captures semantic similarity between phrases  
- Produces high-quality embeddings quickly  

---

### 📊 2.2 Exploring the Data — Why Use Machine Learning  

📁 **File:** `Why_ML_Model.ipynb`  
🔗 [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/Why_ML_Model.ipynb)  

We scaled numeric data, generated text embeddings, and saved the cleaned dataset for exploratory analysis.  

**PCA Analysis:**  
Dimensionality reduction with PCA showed that **50+ components** were required to explain 98% of the variance — too many for practical visualization.  

**KMeans Clustering:**  
We plotted inertia versus cluster number (`k`) to find an “elbow point.”  
However, no clear cluster structure appeared, indicating the need for a more powerful nonlinear model — leading us to use a **Variational Autoencoder (VAE)**.  

---

### 🧮 Part 3: Building the Variational Autoencoder (VAE)

📁 **File:** `model.py`  
🔗 [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/vae_with_kl_annealing/model.py)  

#### Encoder  
Processes numeric features and text embeddings through separate branches:  
- Numeric: 20 → 64 → 32  
- Text: 768 → 128 → 32  

Outputs **μ** and **logσ²**, used to sample latent vector `z` via reparameterization.  

#### Decoder  
Reconstructs both numeric and text inputs from the latent vector:  
- Numeric: latent → 32 → numeric features  
- Text: latent → 128 → 256 → text embeddings  

#### Loss Function  
Total loss = weighted sum of numeric/text reconstruction + KL divergence.  

We used:  
- Weighted numeric/text reconstruction  
- Adjustable KL term  
- **KL annealing** for training stability  

#### Training  
- 100 epochs, batch size 512  
- Early KL warm-up  
- Multiple experiments with varied weights (`WEIGHT_TEXT`, `WEIGHT_NUMERIC`, `KL_WEIGHT`)  
- Saved checkpoints, loss curves, and latent embeddings  

This setup helps capture rich relationships between ingredients, nutrition, and instructions.  

---

### 📈 Part 4: Model Analysis and Recommendations  

📁 **File:** `analysis.ipynb`  

We analyzed training curves and latent distributions to confirm stable training.  

#### Loss Components  

\[
\mathcal{L}_{total} = 
\frac{1}{N} (w_{num} L_{recon}^{num} + w_{text} L_{recon}^{text} + w_{KL} L_{KL})
\]

Increasing each weight affects its corresponding term as expected, confirming interpretable training behavior.  

---

### 🍳 Part 5: Consensus Recommendation Across Models  

📁 **File:** `recommendations_consensus.ipynb`  
🔗 [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/recommendations_consensus.ipynb)  

After training several VAE models, we analyzed their 32-dimensional latent spaces with K-Means and UMAP. Most showed strong structure; one experienced mild posterior collapse.  

To improve reliability, we used a **consensus strategy**:  
1. Generate top-5 similar recipes per model  
2. Aggregate all recommendations  
3. Keep only recipes appearing in ≥3 models  

This consensus ensures stable and interpretable recommendations across models.  

---

### 🍽️ Example Output  

**Liked Recipe**  
**ID:** 326591  
**Name:** Golden Syrup Russian Fudge  

**Instructions (simplified):**  
Mix ingredients (except vanilla) in a saucepan, dissolve sugar, boil until soft-ball stage (120°C), beat until thick and glossy, then cool and enjoy.  

**Recommended Recipes (appearing in ≥3 models):**  

1. **White Caprese Cake (Gluten-Free)** — almond-based cake with white chocolate and lemon zest  
2. **Flaky Oatmeal-Raisin Cookies** — chewy cookies with oats, raisins, and coconut  
3. **$25 Pumpkin Pie** — creamy, spiced pie baked in a buttery crust  
4. **Butter Me Bananas French Toast** — layered French toast with caramelized banana butter  

---

### 🧾 Summary  

By combining multiple VAEs and consensus filtering, we built a **robust recipe recommender** that leverages both numeric nutrition data and semantic text embeddings.  
This approach outperforms simple clustering and creates interpretable, human-meaningful recipe suggestions.

---
