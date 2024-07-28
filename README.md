# Chicago-Crime-Analysis-Spark

## Goal
Analysis Data on Crime in Chicago since 2001. The Analysis is done in Spark.

## 1. Investiage Average Crime

## 2. Analyze spatial Crime differences and highlight hotspots

## 3. Initial Model to predict crime occurances

## 4. Identify temporal patterns of crime

## Bias/Fairness Analysis DATA

I want to analyze my approach based on bias/fairness considerations.
which would be 2 exogenous features I could add to the dataset that will enhance bias/fairness analysis?

### Result:
#### Median Household Income:

- Reason: Income levels can provide insight into economic disparities across different areas. Lower-income areas may face higher crime rates due to various socio-economic factors. Including this feature can help identify whether your model is disproportionately predicting higher crime rates in low-income areas, which could indicate bias.
- Implementation: You can obtain median household income data at a granular level (e.g., census tract or neighborhood) and merge it with your existing dataset based on geographic indicators like Beat or Community Area.

#### Unemployment Rate:

- Reason: Employment status is a critical socio-economic factor influencing crime. Higher unemployment rates may correlate with higher crime rates, and including this feature can help assess if the model is unfairly attributing higher crime risks to areas with higher unemployment, which might reflect socio-economic biases.
- Implementation: Similar to income data, unemployment rate data can be sourced from local labor statistics or census data. Integrate this data at a relevant geographic level to ensure it aligns with your existing spatial features.

### Follow up analysis
based on this dataset, which ones are the protected attributes? are there any attributes in the dataset that could serve as proxies for the protected attributes? What types of fairness analysis can be conducted on this dataset?

### Result:
Based on the dataset, the protected attributes are:

#### Race/Ethnicity:
White
Hispanic
Black
Asian
Mixed
Other

#### Socio-economic Status:
FoodStamps
food_stamp_perc

#### Potential Proxy Attributes
Certain attributes could serve as proxies for protected attributes, which means they can indirectly reflect characteristics of protected groups:

- Population Density: Higher density areas might correlate with certain racial/ethnic groups.
- Median Household Income (if added): Lower income areas might disproportionately represent certain racial/ethnic groups.
- Unemployment Rate (if added): Higher unemployment areas might disproportionately represent certain racial/ethnic groups.
