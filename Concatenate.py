#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:06:19 2018

@author: taniafolabi
"""
import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

quandl.ApiConfig.api_key = '2ubusi8ESqjgTzkEppuD'
data = quandl.get_table('SMA/FBD', brand_ticker='MCD')
#print (data)

# newFans = f(reach, impression, engagementscore)
# for each parameter, i.e newFans = f(reach), newFans = f(impression)
# we see how the scatter plot is for each graph with (matplotlib)
# form order (linear, quadratic)

#print(data.columns)

data1 = data[['date', 'admin_post_reach', 'admin_post_impressions','engagement_score','new_fans']]
#print(data1)

set_of_dates1 = set(data1['date'])
#print('set of dates 1:', set_of_dates1)

#plt.plot(data1.admin_post_reach, data1.engagement_score, 'b.')
data2 = quandl.get_table('SHARADAR/SF1', ticker='MCD')
data2 = data2[['datekey','capex','ebitda']]
#print(data2)

set_of_dates2 = set(data2['datekey'])
#print('set of dates 2:', set_of_dates2)

set_of_dates = set_of_dates1.intersection(set_of_dates2)

dates = np.array(set_of_dates)

engagement_scores = np.array([])
capexs = np.array([])

#print('engagement score    LLL:', type(data1))
#print('shape:', data1['date'].shape)
#engagement_score_dict = dict(data1.lookup('date', 'engagement_score'))
#print('capex:', type(data2))
#capex_dict = dict(data2.lookup('date', 'capex'))
newdata1 = data1.loc[data1['date'].isin(set_of_dates)]
print (newdata1)

newdata2 = data2.loc[data2['datekey'].isin(set_of_dates)]
print (newdata2)

bigdata = pd.concat([newdata1, newdata2], axis=1, sort=False)
print (bigdata)
    
#for d in np.ndenumerate(dates):
#    np.append(engagement_scores, engagement_score_dict[d])
#    print('engagement score added:', engagement_score_dict[d])
#    np.append(capexs, capex_dict[d])
#    print('capex added:', capex_dict[d])

#plt.plot(engagement_scores, capexs,'b.')
#plt.plot(data1.fan_post_count, data2.capex,'b.')



