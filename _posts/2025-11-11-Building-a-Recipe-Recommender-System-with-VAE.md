# Recipe Recommender System with VAE

### Introduction 

The reviews dataset contains **1,401,982 reviews** from **271,907 users**, providing details like author, rating, review text, and more.

### Workflow Overview

The steps for building the recommender system are as follows:

1. **Data Download and Cleaning**  
   We first download the datasets and clean them, handling missing values and inconsistencies.


2. **Preprocessing**  
   - Scale numeric data  
   - Create vector embeddings for textual data such as ingredients and instructions
   - Why did we decided to use VAE


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

### Part 2: Selecting Important Features and Handling Outliers

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

### 2.1 Text Embeddings with all-MiniLM-L6-v2

Before performing dimensionality reduction or building our model, we need to convert the unstructured text from recipe names, ingredients, and instructions into a numerical format that a machine learning model can understand. This process is called **text embedding**. For this, we use the **all-MiniLM-L6-v2** pre-trained model.

##### What is MiniLM?

`all-MiniLM-L6-v2` is a compact, pre-trained language model designed to generate high-quality sentence embeddings. It is part of a family of efficient models that are much faster and smaller than the original BERT models, while still maintaining competitive accuracy.

##### Why we chose this model

- **Efficiency**: MiniLM is a distilled version of a larger language model. It runs significantly faster and requires much less memory, making it practical for large-scale systems without losing too much performance.  
- **High Performance**: Despite its smaller size, MiniLM effectively captures semantic meaning. For example, it understands that *"diced tomatoes and basil"* and *"chopped tomatoes with fresh herbs"* are conceptually similar.  
- **Vector Output (384 Dimensions)**: Each recipe text input is transformed into a dense 384-dimensional vector, providing a high-fidelity numerical representation of the recipe's content and complexity.

### 2.2 Exploring the Data & Why Use a Machine Learning Model

After scaling the numeric data and generating vector embeddings for the text data, we saved the cleaned dataset and started analyzing it to see if we could cluster recipes based on their input features and potentially begin recommending similar recipes.  

All of this analysis can be found in the notebook:  
📁 **File:** `Why_ML_Model.ipynb`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/Why_ML_Model.ipynb)  

In the notebook, we performed **PCA analysis** to reduce dimensionality. By looking at the explained variance, it became clear that we would need **more than 50 principal components** to capture 98% of the variance in the data.  

Next, we attempted to estimate the number of clusters for a **KMeans** algorithm by plotting the inertia (sum of squared distances to the nearest cluster center) for different values of `k`. The “elbow method” helps identify where adding more clusters stops significantly reducing inertia.  

Unfortunately, no meaningful structure emerged from this analysis. This motivated us to move on to a **Variational Autoencoder (VAE)** approach, which could learn a more expressive latent representation of the recipes and enable more effective recommendations.


### Part 3: Building the Variational Autoencoder (VAE) for Recipe Embeddings

After exploring the dataset and seeing that simple clustering did not reveal meaningful structure, we turned to a **Variational Autoencoder (VAE)** to learn a more expressive latent representation of recipes.

The architecture we used consists of **three main components**:
📁 **File:** `model.py`  
🔗 **Source:** [View on GitHub](https://github.com/F-Bafti/VAE-recipe-recommender/blob/main/vae_with_kl_annealing/model.py) 

#### 3.1 Encoder
The encoder takes two types of inputs:

- **Numeric features** (like calories, fat, protein, etc.)
- **Text embeddings** (generated from recipe names, ingredients, and instructions using `all-MiniLM-L6-v2`)

Each input branch is processed separately through fully connected layers:

- Numeric branch: 20 → 64 → 32  
- Text branch: 768 → 128 → 32  

The outputs are concatenated and transformed into two vectors: **mu** and **log variance**, representing the parameters of the latent Gaussian distribution. We then use the **reparameterization trick** to sample a latent vector `z`.

#### 3.2 Decoder
The decoder reconstructs the original inputs from the latent vector:

- Numeric decoder: latent → 32 → numeric features  
- Text decoder: latent → 128 → 256 → text embeddings  

By reconstructing both numeric and text data, the VAE learns a compact, meaningful representation of recipes in the latent space.

#### 3.3 Loss Function
The model is trained using a combination of:

- **Reconstruction Loss** (Mean Squared Error for numeric and text inputs)  
- **KL Divergence** (ensures the latent space approximates a Gaussian distribution)

We also added **weights** to balance numeric and text reconstruction, and an adjustable KL term to control regularization.  

This setup allows the VAE to capture complex relationships between recipe features that simpler models cannot, enabling effective recipe recommendations based on similarity in the learned latent space.

#### 3.4 Training the VAE

Once the VAE architecture was defined, the next step was training the model on our preprocessed recipe data. Here's how we approached it:

**Data Preparation**
We loaded the numeric and text embeddings from our cleaned dataset, split them into **train, validation, and test sets**, and created PyTorch dataloaders. This ensures the model sees a balanced sample during training and allows us to track performance on unseen data.

**Hyperparameter Experiments**
We ran several experiments to find the best weighting between numeric and text reconstructions, as well as the appropriate KL divergence regularization. Key parameters include:

- **`WEIGHT_TEXT`**: Controls importance of reconstructing text embeddings  
- **`WEIGHT_NUMERIC`**: Controls importance of reconstructing numeric features  
- **`KL_WEIGHT`**: Controls the strength of the KL divergence term  

We experimented with a range of values to observe how they influence the latent space quality.

**KL Annealing**
To stabilize training, we used **KL annealing**, gradually increasing the KL weight over the first few epochs. This allows the network to first focus on reconstructing the input well before regularizing the latent space.

**Training Loop**
For each configuration:

1. Initialize a new VAE and optimizer  
2. For each epoch:
   - Forward pass through the encoder and decoder  
   - Compute the **weighted reconstruction loss** and **KL divergence**  
   - Backpropagate and update model weights  
   - Track metrics for numeric/text reconstruction and KL loss  
3. Validate on the hold-out validation set  

Training was performed over **100 epochs** with a **batch size of 512**, which gave the model enough iterations to learn meaningful latent representations.

**Saving Results**
After each experiment, we saved:

- **Model checkpoints**  
- **Training loss history**  
- **Final latent embeddings** for the test set  

These embeddings are later used for **recipe recommendation**, allowing us to find similar recipes in the learned latent space efficiently.

By combining careful preprocessing, weighted loss functions, and KL annealing, our VAE effectively captures both numeric and textual information from the recipes, producing a robust and meaningful latent representation.

### Part 4: Analysis of Model Output and Recommendations:

After training all the experiments, we analyzed the results in `analysis.ipynb` by looking at **loss curves** and latent distributions to ensure the models were behaving correctly.

#### VAE Loss Function

The model is trained using a **weighted VAE loss**, combining three components:

$$
\mathcal{L}_{\text{total}} = 
\frac{1}{N} \Big(
w_{\text{num}} \cdot \mathcal{L}_{\text{recon}}^{\text{num}} +
w_{\text{text}} \cdot \mathcal{L}_{\text{recon}}^{\text{text}} +
w_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}
\Big)
$$

Where:  

- $\mathcal{L}_{\text{recon}}^{\text{num}} = \|x_{\text{num}} - \hat{x}_{\text{num}}\|^2$  
- $\mathcal{L}_{\text{recon}}^{\text{text}} = \|x_{\text{text}} - \hat{x}_{\text{text}}\|^2$  
- $\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$  
- $w_{\text{num}}, w_{\text{text}}, w_{\text{KL}}$ are the numeric, text, and KL weights respectively  
- $N$ is the batch size

#### Interpretation

From the loss plots:  

- Increasing **`numeric_weight`** ($w_{\text{num}}$) reduces numeric reconstruction loss  
- Increasing **`text_weight`** ($w_{\text{text}}$) reduces text reconstruction loss  
- Increasing **`kl_weight`** ($w_{\text{KL}}$) decreases KL divergence loss  

These trends confirm that the model behaves as expected — each loss component responds appropriately to its corresponding weight, demonstrating **stable and interpretable training dynamics**.

We also visualized the **latent distributions** of $\mu$ and $\log \sigma^2$ for all models. Only one configuration showed a minor **posterior collapse**, but the majority of models maintained meaningful latent space representations, validating the effectiveness of our VAE setup.

### Generating Recipe Recommendations from the Latent Space

After training multiple VAE models, we used the resulting **32-dimensional latent embeddings** to generate recipe recommendations. To visualize the latent structure, we applied **KMeans clustering** and **UMAP** for dimensionality reduction to 2D. Most models produced well-organized latent spaces, except for one model that showed a weak posterior collapse and, consequently, a less meaningful representation.

For recommendations, we adopted a **consensus-based approach**:  

1. Select a random set of recipes from the test set.  
2. For each recipe, retrieve the top similar recipes from each trained model using clustering in the latent space.  
3. Count how many models recommend each recipe.  
4. Only suggest recipes that appear in at least a defined threshold of models (in this case, 3).  

This strategy reduces noise and improves the reliability of recommendations. The process ensures that only recipes consistently identified as similar across multiple models are suggested.  

The output is presented clearly for each selected recipe: the original recipe details, followed by the recommended recipes that meet the consensus threshold. This method leverages both numeric and text features encoded in the latent space, providing **robust and interpretable recommendations** based on a combination of learned representations rather than raw features alone.

### Some Recommendation Examples:
======================================================================
🍽️ Liked Recipe
ID: 326591
Name: Golden Syrup Russian Fudge
Instructions:
c("Place all the ingredients except the vanilla, into a medium-heavy saucepan.", "Warm over a gentle heat until the sugar has dissolved.", "Bring to a gentle boil and cook for about 15 — 20 minutes stirring occasionally, until it reaches the soft ball stage (120°C).", "Remove from the heat and add the vanilla.", "Beat (I use an electric mixer) until the fudge is creamy and thick and has lost its gloss for about 4 minutes.", "Pour into a greased 20 cm cake pan.", "Allow to cold and harden.", "Enjoy!"
)

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 518256
🍲 Name: &ldquo;White Caprese&quot; Cake Gluten Free
📖 Instructions:
c("Blitz in the mixer almonds till very small pieces but not too thin (like almond flour). Set aside.If you have almonds with skin, boil them for 5 minutes to peel them easily.", "Melt chopped white chocolate in microwave oven or in a small pot over a bigger one  with boiling water (bain-marie). Grate lemon skin into a small cup and set aside.", "Mix eggs with sugar and add the lemon zest and the soft butter. Keep mixing, add tepid chocolate, almonds, limoncello, the sift flour and the baking powder at the end.", 
"Butter and flour a 28/30 cm cake tin, pour in the cake mixture. Bake  in the preheated oven at 170°C for 40 minutes.", "NB: pay attention to the face of the cake that burns easily. After 20 minutes passed, put a tinfoil over the cake and keep cooking.", "Cool and sprinkle with powdered sugar.")

----------------------------------------------------------------------
📝 Recipe ID: 216030
🍲 Name: &quot; Flaky&quot; Oatmeal-Raisin Cookies
📖 Instructions:
c("In a large saucepan, combine brown sugar and butter.  Cream until well blended.", "Stir in eggs and mix well.", "Combine baking soda, baking powder, salt and flour.  Fold into sugar-butter mixture.", "Stir in vanilla, raisins, oats, corn flakes, coconut and walnuts.  Blend well.", "Scoop out balls of cookie dough with an ice cream scoop or tablespoon onto cookie sheets.", "Flatten each scoop slightly with palm or glass.", "Bake at 350° for 13-15 minutes.")

----------------------------------------------------------------------
📝 Recipe ID: 17265
🍲 Name: $25 Pumpkin Pie
📖 Instructions:
c("Adjust an oven rack to lowest position and heat oven to 400°F.", "Prepare pie crust for blind baking (prick bottom with a fork, line crust with parchment and fill with dried rice or beans). Bake crust for 15 minutes.", "(Have all your ingredients measured out and start your filling when the crust goes into the oven).  Remove parchment from crust,  prick any bubbles with a sharp fork and bake shell for 8 to 10 minutes longer, or until bottom just begins to color.", "Remove crust from oven and brush lightly with egg white while still hot.", 
"Process pumpkin, sugar, spices and salt ingredients in a food processor fitted with steel blade for 1 minute.", "Transfer mixture to a 3-quart heavy saucepan.", "Bring it to a sputtering simmer over medium-high heat.", "Cook pumpkin, stirring constantly, until thick and shiny, about 5 minutes.", "As soon as pie shell comes out of oven, whisk heavy cream and milk into pumpkin and bring to a bare simmer.", "Process eggs in food processor until whites and yolks are mixed, about 5 seconds.", "Gradually pour hot pumpkin mixture through feed tube while still running.", 
"Process 30 seconds.", "Pour warm filling into hot pie shell.", "Put a crust shield on completed pie and bake about 25 minutes.", "Filling will be dry-looking and lightly cracked around the edge.", "The center will wiggle when pie is gently shaken.", "Cool on a wire rack to a warm room temperature.", "Serve with slightly sweetened whipped cream.")

----------------------------------------------------------------------
📝 Recipe ID: 188928
🍲 Name: &quot; Butter Me Bananas&quot; French Toast
📖 Instructions:
c("TIP: it's always good to read ahead.", "Lightly spray large skillet or a griddle & put on 350°F or medium-high.", "In a square  dish, or Tupperware container, using fork or whisk, mix together egg, milk, vanilla, and allspice.", "Flip bread in mixture one at a time until both sides are coated (there should be extra coating left), then lay in pan.", "While bread is cooking, break 1/3 of the banana off and put in a microwave safe bowl(I use a coffee cup), and add the butter. Microwave for about 30 seconds on high, then mash and stir with fork.  Put in for another 30 seconds - 1 minute (it should be very mushy) stir.", 
"CHECK BREAD! Flip to other side and pour remainder of egg coating evenly on tops of bread. Push around with spatula to evenly cover top, when it goes over the edge and spreads into the pan just squish to side of bread. Sprinkle top with cinnamon and flip when bottom has browned. Let brown.", "Place french toast on plate and spread \"banana butter\" on one slice and place other slice on top. Cut.", "Put the syrup into microwave safe bowl and microwave for 15 seconds or until warm and very thin. Drizzle over bread. Split the rest of the banana in half. Place one half above and one to the side of your french toast. (Serve, eat warm) Enjoy --  I sure do :)."
)

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 298141
Name: Canadian Spaghetti Carbonara
Instructions:
c("Cook spaghetti in boiling salted water. While pasta is cooking, heat olive oil in a skillet. Add chopped onion and minced garlic to the skillet.  Chop cooked bacon and add to skillet as well.  Chop red pepper into small pieces and add to skillet.  (Works better with smaller pieces.)  Sprinkle generously with pepper.  Cook until onion is soft (or a bit longer if you prefer).", "Drain cooked spaghetti and return to pot over a medium heat.  Quickly add beaten eggs and toss well until noodles are coated.  Add parmesan and skillet mixture, tossing until mixed well.", 
"Move to pasta bowl and top with more shredded parmesan.", "Enjoy!")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 33606
🍲 Name: "Italian Sandwich" Pasta Salad
📖 Instructions:
c("Cook pasta and set aside.", "Place onions, dill pickles, olives, water chestnuts, chives and peppers in a bowl.", "Add oil, vinegar and spices.", "Marinate for 15-20 minutes or so.", "Add pasta,tomatoes, meat and cheese and toss lightly.", "Serve cold.")

----------------------------------------------------------------------
📝 Recipe ID: 90921
🍲 Name: "I Stole the Idea from Mirj" Sesame Noodles
📖 Instructions:
c("In a large pot, cook your angel hair pasta until almost al dente; drain well, put in a bowl and set aside.", "In a small tupperware container with a lid, mix sesame oil, soy sauce, honey, garlic and sesame seeds.", "Put the lid on the container and shake, shake, shake it like a polaroid picture.", "Pour the sauce over the angel hair and toss well; the heat of the pasta will cook the garlic a bit so it won't be totally raw.", "Mix in the green onions and stir fry veggies.", "Serve.")

----------------------------------------------------------------------
📝 Recipe ID: 105609
🍲 Name: "Salmon" Janet Evening Salad
📖 Instructions:
c("Cook Macaroni AL DENTE.", "Drain and rinse with cold water.", "In a large bowl, combine macaroni, cheese, salmon, relish and onion.", "In a separate bowl, stir together the salad dressing, salt, pepper and garlic.", "pour over macaroni mixture and stir to combine.", "Cover and let chill for at least 3 hours.", "Makes 6 to 8 servings.", "Can be doubled.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 207002
Name: Nacho Cheese Sauce
Instructions:
c("Melt butter.", "Saute onions until caramelized.", "Add flour and cook for 2-3 minutes on medium heat.", "Add milk and cheese and stir frequently until melted, about 15 minutes.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 105609
🍲 Name: "Salmon" Janet Evening Salad
📖 Instructions:
c("Cook Macaroni AL DENTE.", "Drain and rinse with cold water.", "In a large bowl, combine macaroni, cheese, salmon, relish and onion.", "In a separate bowl, stir together the salad dressing, salt, pepper and garlic.", "pour over macaroni mixture and stir to combine.", "Cover and let chill for at least 3 hours.", "Makes 6 to 8 servings.", "Can be doubled.")

----------------------------------------------------------------------
📝 Recipe ID: 42522
🍲 Name: "The Man's" Taco Dip
📖 Instructions:
c("Mix cream cheese and sour cream until a soft mixture.", "Add taco seasoning and bean dip.", "Add 3/4 of cheese and pour into a baking dish.", "Add remaining cheese on top.", "Bake at 350 deg for 20 to 30 minutes or until golden brown.", "Serve with torillla chips.")

----------------------------------------------------------------------
📝 Recipe ID: 53402
🍲 Name: "KILLER" Lasagna
📖 Instructions:
c("Brown the sausage and ground meat and drain all the fat.", "Add all the other ingredients for the meat filling and simmer on the stove, uncovered, for about 1/2 hour.", "I usually add a bay leaf or two.", "Mix up the cheese filling and set aside in fridge until ready to use.", "Put the noodles in salted water and cook until soft when poked with a fork.", "When the noodles are done, drain the noodles and rinse off with cold water so they can be handled easily.", "Spray the baking pan with a non-stick spray and put a layer of noodles down on the bottom of the pan.  It is OK to cut some of the noodles to fit into the pan if they do not fit whole.", 
"Spread half the cheese filling over the noodels, then half of the mozarella cheese on top of that, then half of the meat sauce over the grated mozarella cheese.", "REPEAT layers and then top with more mozarella cheese.", "Bake at 375 degrees for 30 minutes.  Let stand for 10 minutes before cutting into it.", "Leftovers can be easily stored in the freezer.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 243299
Name: Carrot Cake Ice Cream
Instructions:
c("Cook about 1 1/4 cup sliced fresh carrots in the microwave or stovetop until soft.  Blend in food processor or blender with enough water or milk to make a puree.  Measure out 3/4 cup carrot puree, set aside.", "Heat the 1 cup of heavy cream with the brown sugar in a medium size saucepan on medium heat, stirring often.", "When cream mixture is hot, whisk vigorously while adding the gelatin powder.  Add in raisins (only if using an ice cream maker) and cream cheese and continue to heat for about 4 minutes, stirring often.", 
"Remove saucepan from heat.  Pour hot mixture into a large glass bowl.  Add maple syrup, cinnamon, vanilla, and salt. Whisk in evaporated milk (or heavy cream or half and half), and pureed carrot.", "Chill in fridge for about 1 hour or 1/2 hour in freezer, till ice cream mixture is cold.  The ice cream mixture will be thicker at this point.", "ICE CREAM MAKER: Follow manufacturer’s directions, adding chopped pecans near the end of the freezing process.", "FREEZER METHOD: *Note - this method will take longer than the listed time to make*.  Skip the previous chill step. Add in raisins and pecans. Pour ice cream mixture in a covered shallow pan and freeze until almost solid.  Break up frozen mixture and process in food processor or blender until soft (this incorporates air into the ice cream).  Repeat 1 to 2 more times."
)

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 58651
🍲 Name: "Turtle" Squares
📖 Instructions:
c("Preheat oven to 350 degrees F.", "Spray a 13 X 9 baking pan evenly with non-stick cooking spray.", "Beat 1 cup brown sugar with 1/2 cup melted butter with an electric mixer on medium for 2-3 minutes.", "Add the flour mixture and mix until smooth.", "Press the flour mixture evenly and firmly into the prepared baking pan.", "Sprinkle the pecans evenly oven the flour mixture in the pan.", "Mix the 2/3 cup butter and 1/2 cup brown sugar in a saucepan and bring to a boil over a medium heat.", "Stir constantly while boiling for about 1 minute.", 
"Spread the boiling mixture evenly over the pecans.", "Bake for about 20 minutes.", "Cool.", "Melt the chocolate chips in the microwave, stirring ever 30 seconds or so until almost smooth (you can do this in a double boiler on the stove if you prefer).", "Remove the chocolate chips, stir thoroughly to smooth and spread evenly over the top of the squares.", "Chill for 15 minutes until the chocolate sets and then remove from fridge.", "When squares have returned to room temperature (so chocolate won't crack) cut into 36 squares."
)

----------------------------------------------------------------------
📝 Recipe ID: 518256
🍲 Name: &ldquo;White Caprese&quot; Cake Gluten Free
📖 Instructions:
c("Blitz in the mixer almonds till very small pieces but not too thin (like almond flour). Set aside.If you have almonds with skin, boil them for 5 minutes to peel them easily.", "Melt chopped white chocolate in microwave oven or in a small pot over a bigger one  with boiling water (bain-marie). Grate lemon skin into a small cup and set aside.", "Mix eggs with sugar and add the lemon zest and the soft butter. Keep mixing, add tepid chocolate, almonds, limoncello, the sift flour and the baking powder at the end.", 
"Butter and flour a 28/30 cm cake tin, pour in the cake mixture. Bake  in the preheated oven at 170°C for 40 minutes.", "NB: pay attention to the face of the cake that burns easily. After 20 minutes passed, put a tinfoil over the cake and keep cooking.", "Cool and sprinkle with powdered sugar.")

----------------------------------------------------------------------
📝 Recipe ID: 27087
🍲 Name: "Get the Sensation" Brownies
📖 Instructions:
c("Preheat oven to 350 degrees.", "Grease 13 x 9 baking pan.", "In a large bowl, whisk together butter, sugar and vanilla.", "Add eggs and stir until well combined.", "Combine dry ingredients and blend well with wet ingredients.", "Reserve 2 cups of batter and set aside.", "Spread remaining batter in prepared pan.", "Arrange Peppermint Patties in a single layer over batter, about 1/2 inch apart.", "Carefully spread reserved 2 cups of batter on top.", "Bake for 50- 55 minutes or until brownies begin to pull away from sides of pan.", 
"Cool completely on wire rack and cut into squares.")

----------------------------------------------------------------------
📝 Recipe ID: 216030
🍲 Name: &quot; Flaky&quot; Oatmeal-Raisin Cookies
📖 Instructions:
c("In a large saucepan, combine brown sugar and butter.  Cream until well blended.", "Stir in eggs and mix well.", "Combine baking soda, baking powder, salt and flour.  Fold into sugar-butter mixture.", "Stir in vanilla, raisins, oats, corn flakes, coconut and walnuts.  Blend well.", "Scoop out balls of cookie dough with an ice cream scoop or tablespoon onto cookie sheets.", "Flatten each scoop slightly with palm or glass.", "Bake at 350° for 13-15 minutes.")

----------------------------------------------------------------------
📝 Recipe ID: 17265
🍲 Name: $25 Pumpkin Pie
📖 Instructions:
c("Adjust an oven rack to lowest position and heat oven to 400°F.", "Prepare pie crust for blind baking (prick bottom with a fork, line crust with parchment and fill with dried rice or beans). Bake crust for 15 minutes.", "(Have all your ingredients measured out and start your filling when the crust goes into the oven).  Remove parchment from crust,  prick any bubbles with a sharp fork and bake shell for 8 to 10 minutes longer, or until bottom just begins to color.", "Remove crust from oven and brush lightly with egg white while still hot.", 
"Process pumpkin, sugar, spices and salt ingredients in a food processor fitted with steel blade for 1 minute.", "Transfer mixture to a 3-quart heavy saucepan.", "Bring it to a sputtering simmer over medium-high heat.", "Cook pumpkin, stirring constantly, until thick and shiny, about 5 minutes.", "As soon as pie shell comes out of oven, whisk heavy cream and milk into pumpkin and bring to a bare simmer.", "Process eggs in food processor until whites and yolks are mixed, about 5 seconds.", "Gradually pour hot pumpkin mixture through feed tube while still running.", 
"Process 30 seconds.", "Pour warm filling into hot pie shell.", "Put a crust shield on completed pie and bake about 25 minutes.", "Filling will be dry-looking and lightly cracked around the edge.", "The center will wiggle when pie is gently shaken.", "Cool on a wire rack to a warm room temperature.", "Serve with slightly sweetened whipped cream.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 106167
Name: Ginny Dunand's Triple Good Salad
Instructions:
c("STIR 1-1/2 cups of the boiling water into lime Jell-O in medium bowl at least 2 minute until completely dissolved.", "Add 1 cup of the ice cubes; stir until ice is completely melted.", "Pour into 13x9-inch dish.", "Refrigerate 20 minute or until gelatin is set but not firm.", "Stir in crushed pineapple and pecans.", "Refrigerate until firm.", "Using cold beaters and mixing bowl, whip together the whipping cream and cream cheese until fluffy.", "Spread on top of Jell-O.", "Chill until firm.", "Mix together in top of double boiler, sugar, reserved pineapple juice, lemon juice, flour and eggs.", 
"Cook until thick.", "Let cool, the spread on top of other layers.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 111875
🍲 Name: "The Elvis" Smoothie
📖 Instructions:
c("Place all ingredients in blender.", "Blend until very smooth.", "Serve immediately.", "For a vegan smoothie, make with soy milk and soy yogurt.")

----------------------------------------------------------------------
📝 Recipe ID: 23933
🍲 Name: "Chinese" Candy
📖 Instructions:
c("Melt butterscotch chips in heavy saucepan over low heat.", "Fold in peanuts and chinese noodles until coated.", "Drop by tablespoon onto waxed paper.", "Let stand in cool place until firm.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 265509
Name: Tawney's Tex-Mex Chicken and Zucchini
Instructions:
c("Prepare rice first. Open rotel and drain the liquid into a 2 cup measuring cup. You'll be using the rotel when the chicken is cooked so put it aside for now. Add the water to the measuring cup that contains the rotel liquid to equal 2 cups total. Pour into a sauce pan and add the rice. Rice is easy to cook if you do 2 things. Let it come to a boil then turn it down to low and leave it alone. Place a lid on the pot, slightly ajar. Do not lid it tight or it will over flow. Cook for 15 minutes. Turn off the heat, lid tightly and let the rice sit till ready to use.", 
"While the rice cooks, slice the zucchini. I like to slice it thin and slightly oval. Slice the chicken breast in about 1/2 inch slices. Choose a large skillet and add the oil, chicken, seasoning salt and pepper, stirring to coat the chicken evenly. Cook on medium to medium high for about 10 minutes or until the chicken is cooked through but not overcooked. Add the zucchini and rotel put a lid on it and turn to low. Cook for 5 minutes or until the zucchini is lightly cooked- still somewhat crunchy- Do not overcook AND take off the lid when done to ensure it doesn't continue to cook. Adjust seasoning to taste."
)

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 33606
🍲 Name: "Italian Sandwich" Pasta Salad
📖 Instructions:
c("Cook pasta and set aside.", "Place onions, dill pickles, olives, water chestnuts, chives and peppers in a bowl.", "Add oil, vinegar and spices.", "Marinate for 15-20 minutes or so.", "Add pasta,tomatoes, meat and cheese and toss lightly.", "Serve cold.")

----------------------------------------------------------------------
📝 Recipe ID: 90921
🍲 Name: "I Stole the Idea from Mirj" Sesame Noodles
📖 Instructions:
c("In a large pot, cook your angel hair pasta until almost al dente; drain well, put in a bowl and set aside.", "In a small tupperware container with a lid, mix sesame oil, soy sauce, honey, garlic and sesame seeds.", "Put the lid on the container and shake, shake, shake it like a polaroid picture.", "Pour the sauce over the angel hair and toss well; the heat of the pasta will cook the garlic a bit so it won't be totally raw.", "Mix in the green onions and stir fry veggies.", "Serve.")

----------------------------------------------------------------------
📝 Recipe ID: 105609
🍲 Name: "Salmon" Janet Evening Salad
📖 Instructions:
c("Cook Macaroni AL DENTE.", "Drain and rinse with cold water.", "In a large bowl, combine macaroni, cheese, salmon, relish and onion.", "In a separate bowl, stir together the salad dressing, salt, pepper and garlic.", "pour over macaroni mixture and stir to combine.", "Cover and let chill for at least 3 hours.", "Makes 6 to 8 servings.", "Can be doubled.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 477192
Name: Gluten-Free Quiche With Rice and Lentils
Instructions:
c("In a medium sized bowl, combine cooked rice, melted butter, and one slightly beaten egg. Pour into a buttered 9-inch pie plate, pressing against the sides and bottom to form a crust.", "In frying pan, slowing carmelize sliced onion over low heat. Add peppers for last 5 min of cooking.", "Sprinkle one-half of the grated cheese on the bottom of the rice crust. Layer the vegetables and lentils on the cheese. Sprinkle the rest of the cheese over the vegetables. Sprinkle with chopped parsley.", "Combine eggs, milk, salt and pepper in a small bowl and carefully pour over the quiche filling.", 
"Bake in a 400-degree oven for 30 minutes. Let quiche stand for 5 minutes before serving.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 33606
🍲 Name: "Italian Sandwich" Pasta Salad
📖 Instructions:
c("Cook pasta and set aside.", "Place onions, dill pickles, olives, water chestnuts, chives and peppers in a bowl.", "Add oil, vinegar and spices.", "Marinate for 15-20 minutes or so.", "Add pasta,tomatoes, meat and cheese and toss lightly.", "Serve cold.")

----------------------------------------------------------------------
📝 Recipe ID: 90921
🍲 Name: "I Stole the Idea from Mirj" Sesame Noodles
📖 Instructions:
c("In a large pot, cook your angel hair pasta until almost al dente; drain well, put in a bowl and set aside.", "In a small tupperware container with a lid, mix sesame oil, soy sauce, honey, garlic and sesame seeds.", "Put the lid on the container and shake, shake, shake it like a polaroid picture.", "Pour the sauce over the angel hair and toss well; the heat of the pasta will cook the garlic a bit so it won't be totally raw.", "Mix in the green onions and stir fry veggies.", "Serve.")

----------------------------------------------------------------------
📝 Recipe ID: 59389
🍲 Name: "Alouette" Potatoes
📖 Instructions:
c("Place potatoes in a large pot of lightly salted water and bring to a gentle boil.", "Cook until potatoes are just tender.", "Drain.", "Place potatoes in a large bowl and add all ingredients except the\"Alouette\".", "Mix well and transfer to a buttered 8x8 inch glass baking dish with 2 inch sides.", "Press the potatoes with a spatula to make top as flat as possible.", "Set aside for 2 hours at room temperature.", "Preheat oven to 350^F.", "Spread\"Alouette\" evenly over potatoes and bake 15 minutes.", 
"Divide between plates.", "Garnish with finely diced red and yellow bell peppers.")

----------------------------------------------------------------------
📝 Recipe ID: 105609
🍲 Name: "Salmon" Janet Evening Salad
📖 Instructions:
c("Cook Macaroni AL DENTE.", "Drain and rinse with cold water.", "In a large bowl, combine macaroni, cheese, salmon, relish and onion.", "In a separate bowl, stir together the salad dressing, salt, pepper and garlic.", "pour over macaroni mixture and stir to combine.", "Cover and let chill for at least 3 hours.", "Makes 6 to 8 servings.", "Can be doubled.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 63558
Name: grilled glazed tuna
Instructions:
c("stir together cornstarch and 1 1/2 tbsp water until smooth, set aside.", "add pineapple and 2 tbsp water to blender and puree.", "in a saucepan add pineapple, soy sauce, sugar, ginger, bring to a boil.", "add cornstarch, stir 1 minute.", "heat grill.", "add salt& pepper to tuna, cook on oiled grill, turning once about 5 mins per side, or to taste spread pineapple sauce on each tuna steak and grill,covered, 1-2 mins more, serve mix wasabi powder, water and mayo together, serve alongside tuna.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 105609
🍲 Name: "Salmon" Janet Evening Salad
📖 Instructions:
c("Cook Macaroni AL DENTE.", "Drain and rinse with cold water.", "In a large bowl, combine macaroni, cheese, salmon, relish and onion.", "In a separate bowl, stir together the salad dressing, salt, pepper and garlic.", "pour over macaroni mixture and stir to combine.", "Cover and let chill for at least 3 hours.", "Makes 6 to 8 servings.", "Can be doubled.")

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 245844
Name: Custard Tart With Garibaldi Biscuits by Marcus Wareing
Instructions:
c("Preheat the oven to 180C/350F/Gas 4.", "First, make the biscuits. Mix together the butter, icing sugar and flour until smooth. Slowly add the egg whites, stirring, until they are completely incorporated, then fold in the currants. Bring together into a ball, wrap in cling film and chill for at least one hour.", "Roll the dough on a lightly floured surface out to 5mm/½in thick. Cut into 12 rectangles - you may have some dough left over. Place on a baking tray lined with greaseproof paper, ensuring the biscuits are not touching each other. Chill in the fridge for 30 minutes.", 
"Bake the biscuits for 8-10 minutes or until golden brown. Remove and cool on a wire rack. Keep in an airtight tin.", "For the pastry, rub together the flour, salt, lemon zest and butter until the mixture resembles breadcrumbs. Add the sugar, then beat together the egg yolk and whole egg and slowly add these, mixing until the pastry forms a ball. Wrap tightly in cling film and refrigerate for two hours.", "Turn the oven down to 170C/325F/Gas 3.", "Roll out the pastry on a lightly floured surface to 2mm/one eighth-in thickness. Use to line an 18cm/7in flan ring placed on a baking sheet. Line with greaseproof paper and fill with baking beans, then bake blind for about 10 minutes or until the pastry is starting to turn golden brown. Remove the paper and beans, and allow to cool.", 
"Turn the oven down to 130C/250F/Gas1.", "For the filling, whisk together the yolks and sugar. Add the cream and mix well. Pass the mixture through a fine sieve into a saucepan and heat to blood temperature.", "Fill the pastry case with the custard, until 5mm/¼in from the top. Carefully place in the middle of the oven and bake for 30-40 minutes or until the custard appears set but not too firm. Remove from the oven and cover the surface liberally with grated nutmeg. Allow to cool to room temperature.", 
"Before serving, warm the biscuits through in the oven for 5 minutes. Cut the tart with a sharp knife and serve with the biscuits.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 518256
🍲 Name: &ldquo;White Caprese&quot; Cake Gluten Free
📖 Instructions:
c("Blitz in the mixer almonds till very small pieces but not too thin (like almond flour). Set aside.If you have almonds with skin, boil them for 5 minutes to peel them easily.", "Melt chopped white chocolate in microwave oven or in a small pot over a bigger one  with boiling water (bain-marie). Grate lemon skin into a small cup and set aside.", "Mix eggs with sugar and add the lemon zest and the soft butter. Keep mixing, add tepid chocolate, almonds, limoncello, the sift flour and the baking powder at the end.", 
"Butter and flour a 28/30 cm cake tin, pour in the cake mixture. Bake  in the preheated oven at 170°C for 40 minutes.", "NB: pay attention to the face of the cake that burns easily. After 20 minutes passed, put a tinfoil over the cake and keep cooking.", "Cool and sprinkle with powdered sugar.")

----------------------------------------------------------------------
📝 Recipe ID: 216030
🍲 Name: &quot; Flaky&quot; Oatmeal-Raisin Cookies
📖 Instructions:
c("In a large saucepan, combine brown sugar and butter.  Cream until well blended.", "Stir in eggs and mix well.", "Combine baking soda, baking powder, salt and flour.  Fold into sugar-butter mixture.", "Stir in vanilla, raisins, oats, corn flakes, coconut and walnuts.  Blend well.", "Scoop out balls of cookie dough with an ice cream scoop or tablespoon onto cookie sheets.", "Flatten each scoop slightly with palm or glass.", "Bake at 350° for 13-15 minutes.")

----------------------------------------------------------------------
📝 Recipe ID: 17265
🍲 Name: $25 Pumpkin Pie
📖 Instructions:
c("Adjust an oven rack to lowest position and heat oven to 400°F.", "Prepare pie crust for blind baking (prick bottom with a fork, line crust with parchment and fill with dried rice or beans). Bake crust for 15 minutes.", "(Have all your ingredients measured out and start your filling when the crust goes into the oven).  Remove parchment from crust,  prick any bubbles with a sharp fork and bake shell for 8 to 10 minutes longer, or until bottom just begins to color.", "Remove crust from oven and brush lightly with egg white while still hot.", 
"Process pumpkin, sugar, spices and salt ingredients in a food processor fitted with steel blade for 1 minute.", "Transfer mixture to a 3-quart heavy saucepan.", "Bring it to a sputtering simmer over medium-high heat.", "Cook pumpkin, stirring constantly, until thick and shiny, about 5 minutes.", "As soon as pie shell comes out of oven, whisk heavy cream and milk into pumpkin and bring to a bare simmer.", "Process eggs in food processor until whites and yolks are mixed, about 5 seconds.", "Gradually pour hot pumpkin mixture through feed tube while still running.", 
"Process 30 seconds.", "Pour warm filling into hot pie shell.", "Put a crust shield on completed pie and bake about 25 minutes.", "Filling will be dry-looking and lightly cracked around the edge.", "The center will wiggle when pie is gently shaken.", "Cool on a wire rack to a warm room temperature.", "Serve with slightly sweetened whipped cream.")

----------------------------------------------------------------------
📝 Recipe ID: 188928
🍲 Name: &quot; Butter Me Bananas&quot; French Toast
📖 Instructions:
c("TIP: it's always good to read ahead.", "Lightly spray large skillet or a griddle & put on 350°F or medium-high.", "In a square  dish, or Tupperware container, using fork or whisk, mix together egg, milk, vanilla, and allspice.", "Flip bread in mixture one at a time until both sides are coated (there should be extra coating left), then lay in pan.", "While bread is cooking, break 1/3 of the banana off and put in a microwave safe bowl(I use a coffee cup), and add the butter. Microwave for about 30 seconds on high, then mash and stir with fork.  Put in for another 30 seconds - 1 minute (it should be very mushy) stir.", 
"CHECK BREAD! Flip to other side and pour remainder of egg coating evenly on tops of bread. Push around with spatula to evenly cover top, when it goes over the edge and spreads into the pan just squish to side of bread. Sprinkle top with cinnamon and flip when bottom has browned. Let brown.", "Place french toast on plate and spread \"banana butter\" on one slice and place other slice on top. Cut.", "Put the syrup into microwave safe bowl and microwave for 15 seconds or until warm and very thin. Drizzle over bread. Split the rest of the banana in half. Place one half above and one to the side of your french toast. (Serve, eat warm) Enjoy --  I sure do :)."
)

----------------------------------------------------------------------

======================================================================
🍽️ Liked Recipe
ID: 282207
Name: Salmon Special for Dogs
Instructions:
c("In a medium saucepan, heat undrained salmon, minced sausage, garlic, rice, undrained spinach and about a cup of water.", "Let everything simmer  for about half an hour, adding more water as it is absorbed so the rice becomes quite mushy.", "Cool and refrigerate in covered container, or freeze in 1/4 cup \"plops\" on a cookie sheet.", "Mix one serving with a little less than usual of your dog's usual dry food.")

✅ Recommended Recipes (appearing in at least 3 models):

📝 Recipe ID: 242081
🍲 Name: &Eacute;pinards &aacute; La Reine - Spinach, Queen Style
📖 Instructions:
c("Beat the egg yolks well; beat the egg whites to soft peaks.", "Cook the onion in the butter; add spinach and fry quickly.", "Add flour and milk and cook until thickened.", "Season with salt and pepper and add grated cheese.", "When it begins to boil, remove from heat and add well beaten egg yolks followed by the egg whites.", "Turn into a lightly greased baking dish; place baking dish into a pan of hot water and bake in a preheated 450F oven for ten minutes.", "Garnish with the shrimp and serve while hot."
)

----------------------------------------------------------------------
📝 Recipe ID: 98930
🍲 Name: "steamed" Chicken Cutlets in Packages
📖 Instructions:
c("Preheat oven to 450°F.", "Rinse the chicken breasts, and either cut or pound the chicken to create cutlets of about equal size.", "Tear off squares of aluminum foil (enough to wrap around the chicken and then some) and place each chicken breast on a square.", "Slice the tomato (es) and place a slice or two atop of each breast.", "Drizzle the chicken with the olive oil.", "Sprinkle the basil on top of the chicken breasts evenly.", "Add salt and pepper to each breast.", "Loosely fold up the foil over each cutlet, leaving both room for air space but folded up enough not to allow oil to drip out.", 
"Place all packages in a large baking dish and bake for 20 minutes.", "Serve while still packaged; allow each diner to open his/her package at the table.")

