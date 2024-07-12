# ParisPrediction2024

## Data Preparation

First, we have correctly encoded our dataset in UTF-8. Next, we filtered our data to include only the competitors who have won a medal, whether bronze, silver, or gold, over the years in the Summer Olympic Games. This was done in the file [`data_prep.ipynb`](./data_prep.ipynb).


### Exploratory Data Analysis (EDA) Summary

All this information can be found in the file  [`EDA.ipynb`](.EDA.ipynb)

 Exploratory Data Analysis (EDA), we performed the following steps:

1. **Initial Data Overview**:
   - Loaded the dataset and provided a general overview.
   - Identified that the only missing values were in the `Age` variable.

2. **Data Cleaning**:
   - Removed rows with missing values in the `Age` variable as they corresponded to very old Olympic participations.
   - Converted the `Age` variable from float to int for consistency.

3. **Outlier Detection**:
   - Verified that there were no outliers in the `Year` and `Entry ID` variables.
   - Identified outliers in the `Age` variable and decided to remove participants younger than 13 years old and older than 70 years old.

4. **Data Visualization**:
   - Created various visualizations to understand the data distribution:
     - A pie chart to show the gender distribution of the participants.
     - A horizontal bar chart to display the top 10 cities hosting Olympic events.
     - A horizontal bar chart for the distribution of sports in the dataset.
     - Horizontal bar charts for the top 10 countries with the most gold, silver, and bronze medals won.

5. **Correlation Matrix**:
   - Conducted a correlation matrix analysis for the numeric variables to understand the relationships between them.

6. **Final Data Preparation**:
   - Saved the cleaned and processed data for further analysis.

This comprehensive EDA allowed us to clean the dataset, identify key patterns and trends, and prepare the data for subsequent analysis.
