# AB Testing
AB testing to statistical randomized control trials is one of the most popular ways for business to test: 
new UX features, new version of a product or new version of an algorithm.

In AB testing, you have two groups of users and you give the current vesrion of the product to one group referred to as the
control group and the new improved version of the product to another group reffred to as the experimental group and your aim as 
a data scientist is to understand the effect of the changes in the product on the user experience. You want to be able to say 
if the new changes to the product has made it a better experinece for the users or not. In this way the bussiness can understand if
it is a good idea to implement the new change or just keep it as it is. or test new variations.

## AB testing steps:

1. **Hypothesis desing**
At this point, all the stake holders decide what is being tested. What is the desired outcome and what KPI (key performance Indicator) we are going to influence. For example at this point we are deciding whether changing the color of a botom in the product page, is going to improve the engagment of the costumers.
In this point we need to define the key primary metric for our test. This is the most important part of the test probably. The sucess metric is only one and we can not define more than one for a test as it is going to bias the results analysis. This metric can be for example the conversion rate. The subscription rate or so on ...

conversion rate = (number of conversion / total number of visitors) * 100

Click Trough Rate (CTR) = (number of clicks / total number of impressions) * 100

At this point, also the the null and the alternative hypothesis must be defined together with all the stakeholders

2. **AB test design**
At this point we first do the power analysis and then from the results of the power analysis, we calculate the minimum sample size needed for the experiment and then based on that we define the duration of the test. so in total there are 3 steps in this part.
for the power analysis we have the following steps:

- Determine the power of the test. The probability of correctly rejecting the null hypothesis. or the probability of rejecting the null hypothesis when the null hypothesis is wrong. it is ususally defined as (1-\beta) where $\beta is the probability of making type II error which is basically the probability of NOT rejecting the null hypotheis when the null is actually wrong. So is we have a power of the test equal to 80%, then it means that we are okay if 20% of the time we make a mistake in rejecting the null hypotheis when it is actually wrong. It is common to have a power of the test equal to 80% most of the time.

- Determine the significance level, which is the probability of make type I error (alpha). Which is basically the probability of rejecting the null hypothesis when it is actually true. It is common to pick 5% as the significance level which means we will reject the null hypothesis if the probability of seeing an effect similar or more extreme of what we observed in the experiment is less than 5%. So it means that we allow ourseleves to make mistakes 5% of the time by rejecting null hypothesis when it is actually correct. It is common to choose 5% as the significance level.

- Determine the minimum detectable effect (delta). A translation from statistical significance to practical significance. For example, even if we see an effect, we should decide how much of an effect is worthy to try to implement for the business. maybe it is effective but does not worht to implelment at the end. this should be dicided tohether with the stakeholders.

  ```
  POWER ANALYSIS:
  
  - Beta: Probability of type two error

  - (1-Beta): Power of the test

  - Alpha:  Probability of type one error, significance level

  - Delta: Minimum detectable effect
   
  ```

4. **Run AB test**
   
   - Resutls analysis
   - Draw conclution
