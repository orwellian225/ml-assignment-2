# Stats Script

Calculate the summary statistics of the specified feature

## Arguments

* df - data_filepath - The file path to the data file
* lf - labels_filepath - The file path to the labels file
* i - feature - The id of the feature to calculate the summary statistics for. -1 will calculate all the summary statistics

## Methods

* calc_summary_stats

### calc_summary_stats

> Overloaded Method
>
> * Specified Feature
> * Unspecified Feature

Calculates the following statistics:

* Mean
* Median
* Variance
* Standard Deviation
* Minimum
* Maximum
* Range

#### Method Arguments
  
* Feature Matrix - the matrix holding all feature values
* Feature - The ID of the feature to calculate for

#### Specified Feature

Calculate the summary statistics only for the specified feature

#### Unspecified Feature

Calculate the summary statistics for all the features
