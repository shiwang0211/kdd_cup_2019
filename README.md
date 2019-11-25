# KDD Cup 2019
The repository contains the solution to KDD Cup 2019 ML competition. Unfortunately our team was not able to pass Phase 1 (Ranked **108**/1702). Link to competition: [https://dianshi.baidu.com/competition/29/rule](https://dianshi.baidu.com/competition/29/rule)


# Context
This competition (Context-Aware Multi-Modal Transportation Recommendation) requires building a supervised learning to predict users' transportation mode choice based on the plans presented to them on the navigation app. The results are evaluated based on **weighted f1-score**.

# Approach
## Feature Engineering


<table>
  <tr>
   <td><strong>Feature</strong>
   </td>
   <td><strong>Detail</strong>
   </td>
   <td><strong>Type</strong>
   </td>
  </tr>
  <tr>
   <td>transport_mode
   </td>
   <td>Mode Index
   </td>
   <td>cate
   </td>
  </tr>
  <tr>
   <td>num_plans
   </td>
   <td>Number of Plans
   </td>
   <td>int
   </td>
  </tr>
  <tr>
   <td>rank
   </td>
   <td>Order of plan presented in query results
   </td>
   <td>int
   </td>
  </tr>
  <tr>
   <td>lon_diff, lat_diff
   </td>
   <td>Difference of lon and lat between O and D
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>mode_hr_clicks
   </td>
   <td>Percent of clicked mode i when presented in queries during hour j 
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>mode_dow_clicks
   </td>
   <td>Percent of clicked mode i when presented in queries during weekday j 
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>mode_rank_clicks
   </td>
   <td>Percent of clicked mode i when presented as rank j in query results
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>mode_mode_list_clicks
   </td>
   <td>Percent of clicked mode i when the plan set in query result is j
   </td>
   <td>float
   </td>
  </tr>
  <tr>
 
   <td>eta_per_distance
<p>
price_per_distance
<p>
price_per_eta
   </td>
   <td>Eta per distance, Price per distance and Price per eta for each plan
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>price_NA
   </td>
   <td>Whether price information is NA
   </td>
   <td>binary
   </td>
  </tr>
  <tr>
   <td>is_night
   </td>
   <td>Whether it is between 11pm-5am
   </td>
   <td>binary
   </td>
  </tr>
  <tr>
   <td>is_weekend
   </td>
   <td>Whether it is Friday-Sunday
   </td>
   <td>Binary
   </td>
  </tr>
  <tr>
   <td>curr_hour
<p>
curr_weekday
   </td>
   <td>Hour and Weekday
   </td>
   <td>int
   </td>
  </tr>
  <tr>
   <td>o_lng/o_lat/d_lng/d_lat
   </td>
   <td>The lon and lat of O and D
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>max/min/mean/std_distance
<p>
max/min/mean/std_price
<p>
max/min/mean/std_eta
   </td>
   <td>The max/min/mean/std of plans under the same query
   </td>
   <td>float
   </td>
  </tr>
  <tr>
   <td>o/d_nearest_subway
   </td>
   <td>The distance to the nearest subway station from O and D
   </td>
   <td>float
   </td>
  </tr>
</table>

## Algorithm
### [Light GBM](https://github.com/shiwang0211/kdd_cup_2019/blob/master/Feature%20Engineering%20and%20LGB.ipynb)
Hyperpamaters are tuned in an iterative manner based on the model performance on validation and test set. The best set of hyperparameters are listed below.

params = {'boosting_type': 'gbdt',
 'objective': 'binary',
 'num_leaves': 31,
 'learning_rate': 0.05,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.9,
 'bagging_seed': 0,
 'bagging_freq': 1,
 'verbose': 1,
 'seed': 42,
 'reg_alpha':7.5,
 'reg_lambda':2}
 
###  [Deep FM](https://github.com/shiwang0211/kdd_cup_2019/blob/master/Deep-FM.ipynb)
This network structure emphasizes both low and high order feature interactions, and requires minimal feature engineering. 
- Link to paper: [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)
<p align="center">
  <img src="./figure/deepfm.png" width="600" title="Wide & deep architecture of DeepFM">
</p>

The network is built from scratch with Tensorflow and Keras with the following steps:
- Define 1st order Factorization Machine layer
- Define 2nd order Factorization Machine layer
- Define Deep NN layers
- Merge outputs from all layers


# Results
The best result of our team has a weighted f1 score of **0.69693201** on test set.
