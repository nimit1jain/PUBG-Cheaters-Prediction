from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


pubg_data=pd.read_csv("N:\Machine learning\Algorithms\project\pubg_train_V2.csv")

# Lets look at our dataset
# print(pubg_data.shape)
# print(pubg_data.head(10))
# print(pubg_data.info())

# Feature descriptions (From Kaggle)
# DBNOs - Number of enemy players knocked.
# assists - Number of enemy players this player damaged that were killed by teammates.
# boosts - Number of boost items used.
# damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
# headshotKills - Number of enemy players killed with headshots.
# heals - Number of healing items used.
# Id - Player’s Id
# killPlace - Ranking in match of number of enemy players killed.
# killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
# killStreaks - Max number of enemy players killed in a short amount of time.
# kills - Number of enemy players killed.
# longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# matchDuration - Duration of match in seconds.
# matchId - ID to identify match. There are no matches that are in both the training and testing set.
# matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
# rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
# revives - Number of times this player revived teammates.
# rideDistance - Total distance traveled in vehicles measured in meters.
# roadKills - Number of kills while in a vehicle.
# swimDistance - Total distance traveled by swimming measured in meters.
# teamKills - Number of times this player killed a teammate.
# vehicleDestroys - Number of vehicles destroyed.
# walkDistance - Total distance traveled on foot measured in meters.
# weaponsAcquired - Number of weapons picked up.
# winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
# groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# numGroups - Number of groups we have data for in the match.
# maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# Now if you can see the dataset their is no column for cheaters prediction so our intial aim is to find the players who are cheating through data analysis and some domain knowledge. Afterwards we will add a column of cheaters in our orignal dataset and annot the data with respect to cheaters, then we will build our machine learning model to predict whether a player is cheating or not.


# Lets dive deeper into data and check the impurites present

               #--------removing any NaN values-------

print(pubg_data[pubg_data['winPlacePerc'].isnull()])

pubg_data.drop(2744604,inplace=True)


plt.figure(figsize=[25,12])
sns.heatmap(pubg_data.corr(),annot = True,cmap = "BuPu")


# If you have played any Battle Royal game you can understand that killing enemy without moving a single step is nearly not possible so these players might be potential cheaters so we are appending these players in cheaters_data.

#                -----dropping categorical attributes as they are of no use for our prediction-------
pubg_data=pubg_data.drop(['Id','groupId', 'matchId','matchType'],axis=1)
plt.figure(figsize=(22,5))
sns.countplot(data=pubg_data, x=pubg_data['numGroups']).set_title('numGroups')
plt.show()


            #---------using totaldistance attribute to sum up all the distances travelled by the player--------
pubg_data['totalDistance'] = pubg_data['rideDistance'] + pubg_data['walkDistance'] + pubg_data['swimDistance']

pubg_data['potential cheaters']=((pubg_data['kills'] > 0) & (pubg_data['totalDistance'] == 0))

cheaters_data=pubg_data[pubg_data['potential cheaters']==True]
pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)


# chances of breaking the world record of 59 kills in PUBG single match is rare so we are marking those players as potential cheaters.

plt.figure(figsize=(12,4))
sns.countplot(data=pubg_data, x=pubg_data['kills']).set_title('Kills')
plt.show()

pubg_data['potential cheaters']=((pubg_data['kills'] > 59))
cheaters_data=pd.concat([cheaters_data,pubg_data[pubg_data['potential cheaters']==True]])

pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)

# # Killing an enemy from a distance of more than 1KM sounds insane until or unless you get on some vehicle and run away leaving your enemy to die but chances of this are very less so we can consider these players as potential cheaters


plt.figure(figsize=(12,4))
sns.distplot(pubg_data['longestKill'],kde=True,color='orange')
plt.show()

pubg_data['potential cheaters']=((pubg_data['longestKill'] >= 1000))
cheaters_data=pd.concat([cheaters_data,pubg_data[pubg_data['potential cheaters']==True]])

pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)

# in a single match a player can acquire on an average of 10-20 weapons so if someone is using more than 50 weapons then they can be considered as potential cheaters


plt.figure(figsize=(12,4))
sns.distplot(pubg_data['weaponsAcquired'], bins=10)
plt.show()


pubg_data['potential cheaters']=((pubg_data['weaponsAcquired'] >= 50))
cheaters_data=pd.concat([cheaters_data,pubg_data[pubg_data['potential cheaters']==True]])

pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)

# if a player is killing enemy without using a single weapon then their is something fishy about it so putting them in cheaters category will be more safer

pubg_data['potential cheaters']=((pubg_data['weaponsAcquired'] == 0) & (pubg_data['kills']>10))
cheaters_data=pd.concat([cheaters_data,pubg_data[pubg_data['potential cheaters']==True]])

pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)

# on an average players dont use more than 25-30 heals in a single match sp we can put these players in cheaters category


plt.figure(figsize=(12,4))
sns.distplot(pubg_data['heals'], bins=10)
plt.show()

pubg_data['potential cheaters']=((pubg_data['heals'] >=30))
cheaters_data=pd.concat([cheaters_data,pubg_data[pubg_data['potential cheaters']==True]])

pubg_data.drop(pubg_data[pubg_data['potential cheaters']==True].index,inplace=True)


pubg_data=pd.concat([pubg_data,cheaters_data])


pubg_data=pubg_data.drop('winPlacePerc',axis=1)

target=pubg_data['potential cheaters']
features=pubg_data.drop('potential cheaters',axis=1)

x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.3,random_state=0)

model=RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features='sqrt')

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_predtrain=model.predict(x_train)

print("test data accuracy: ", accuracy_score(y_test, y_pred))
print("test data precision score: ", precision_score(y_test, y_pred)) 
print("test data recall score: ", recall_score(y_test, y_pred))
print("test data f1 score: ", f1_score(y_test, y_pred))
print("test data area under curve (auc): ", roc_auc_score(y_test, y_pred))
