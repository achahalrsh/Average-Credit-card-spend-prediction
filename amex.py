
#importing different libraries 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[2]:


Data=pd.read_csv("trainamex.csv")


# In[3]:


Test=pd.read_csv('test_amex.csv')


# In[4]:


#Replacing the value of saving with 0 and credit with 1 in account_type column
Data['account_type'] = np.where(Data['account_type']=='saving', 0, 1)
Test['account_type'] = np.where(Test['account_type']=='saving', 0, 1)


# In[5]:


#Replacing the value of gender male with 0 and female with 1 in gender column
Data['gender'] = np.where(Data['gender']=='M', 0, 1)
Test['gender'] = np.where(Test['gender']=='M', 0, 1)


# In[6]:


#Filling unavailable values with 0 in Data and Test file
Data = Data.fillna(0)
Test = Test.fillna(0)


# In[7]:


# creating new variable total spend in each month from debit and credit in Data and Test file
Data['tot_spend_april']=Data['cc_cons_apr']+Data['dc_cons_apr']
Data['tot_spend_may']=Data['cc_cons_may']+Data['dc_cons_may']
Data['tot_spend_jun']=Data['cc_cons_jun']+Data['dc_cons_jun']

Test['tot_spend_april']=Test['cc_cons_apr']+Test['dc_cons_apr']
Test['tot_spend_may']=Test['cc_cons_may']+Test['dc_cons_may']
Test['tot_spend_jun']=Test['cc_cons_jun']+Test['dc_cons_jun']


# In[8]:


#creating new variable which is ratio of count of credit transactions to total number of transactions
Data['tot_count_april']=(Data['cc_count_apr'])/(Data['cc_count_apr']+Data['dc_count_apr']+0.1)
Data['tot_count_may']=(Data['cc_count_may'])/(Data['cc_count_may']+Data['dc_count_may']+0.1)
Data['tot_count_jun']=(Data['cc_count_jun'])/(Data['cc_count_jun']+Data['dc_count_jun']+0.1)

Test['tot_count_april']=(Test['cc_count_apr'])/(Test['cc_count_apr']+Test['dc_count_apr']+0.1)
Test['tot_count_may']=(Test['cc_count_may'])/(Test['cc_count_may']+Test['dc_count_may']+0.1)
Test['tot_count_jun']=(Test['cc_count_jun'])/(Test['cc_count_jun']+Test['dc_count_jun']+0.1)


# In[9]:


#creating new variable which is 
#ratio of count of credit transactions of a months to count of credit card transaction of that month and next month
Data['total_cc_count']=(Data['cc_count_apr']+Data['cc_count_may']+Data['cc_count_jun'])
Data['total_dc_count']=Data['dc_count_may']+Data['dc_count_may']+Data['dc_count_jun']

Test['total_cc_count']=(Test['cc_count_apr']+Test['cc_count_may']+Test['cc_count_jun'])
Test['total_dc_count']=Test['dc_count_may']+Test['dc_count_may']+Test['dc_count_jun']


# In[10]:


#creating new variable which is average credit spend of april, may and june
Data['avg_three_mnths']=(Data['cc_cons_apr']+Data['cc_cons_may']+Data['cc_cons_jun'])/3

Test['avg_three_mnths']=(Test['cc_cons_apr']+Test['cc_cons_may']+Test['cc_cons_jun'])/3


# In[11]:


#creating new variable which is maximum and minimum credit spend of three months
Data['max_three_mnths']=Data[['cc_cons_apr','cc_cons_may','cc_cons_jun']].max(axis=1)
Data['min_three_mnths']=Data[['cc_cons_apr','cc_cons_may','cc_cons_jun']].min(axis=1)

Test['max_three_mnths']=Test[['cc_cons_apr','cc_cons_may','cc_cons_jun']].max(axis=1)
Test['min_three_mnths']=Test[['cc_cons_apr','cc_cons_may','cc_cons_jun']].min(axis=1)


# In[12]:


# creating new variable which is growth rate of spend from april to may
Data['inc_aprtomay']=(Data['cc_cons_may']-Data['cc_cons_apr'])/(Data['cc_cons_apr']+0.1)

#creating new variable which is growth rate of spend from may to june 
Data['inc_maytojun']=(Data['cc_cons_jun']-Data['cc_cons_may'])/(Data['cc_cons_may']+0.1)

#creating new variable which is growth rate of spend from april to may
Test['inc_aprtomay']=(Test['cc_cons_may']-Test['cc_cons_apr'])/(Test['cc_cons_apr']+0.1)

#creating new variable which is growth rate of spend from may to june 
Test['inc_maytojun']=(Test['cc_cons_jun']-Test['cc_cons_may'])/(Test['cc_cons_may']+0.1)


# In[13]:


#creating new variable which is credit card spend in month divided by count of credit card transactions
Data['totaltonumberoftransapr']=(Data['cc_cons_apr'])/(Data['cc_count_apr']+0.1)
Data['totaltonumberoftransmay']=(Data['cc_cons_may'])/(Data['cc_count_may']+0.1)
Data['totaltonumberoftransjune']=(Data['cc_cons_jun'])/(Data['cc_count_jun']+0.1)

#creating new variable which is credit card spend in month divided by count of credit card transactions
Test['totaltonumberoftransapr']=(Test['cc_cons_apr'])/(Test['cc_count_apr']+0.1)
Test['totaltonumberoftransmay']=(Test['cc_cons_may'])/(Test['cc_count_may']+0.1)
Test['totaltonumberoftransjune']=(Test['cc_cons_jun'])/(Test['cc_count_jun']+0.1)


# In[14]:


#creating new variable which is credit card divided total spend 
Data['crd_to_total_apr']=(Data['cc_cons_apr'])/(Data['tot_spend_april']+0.1)
Data['crd_to_total_may']=(Data['cc_cons_may'])/(Data['tot_spend_may']+0.1)
Data['crd_to_total_jun']=(Data['cc_cons_jun'])/(Data['tot_spend_jun']+0.1)

#creating new variable which is credit card divided total spend 
Test['crd_to_total_apr']=(Test['cc_cons_apr'])/(Test['tot_spend_april']+0.1)
Test['crd_to_total_may']=(Test['cc_cons_may'])/(Test['tot_spend_may']+0.1)
Test['crd_to_total_jun']=(Test['cc_cons_jun'])/(Test['tot_spend_jun']+0.1)


# In[15]:


#creating new variable which is ratio of credit to total of three months 
Data['credit_to_tot_of_all']=(Data['cc_cons_apr']+Data['cc_cons_may']+Data['cc_cons_jun'])/(Data['tot_spend_april']+Data['tot_spend_may']+Data['tot_spend_jun']+0.1)

#creating new variable which is ration of credit to total of three months 
Test['credit_to_tot_of_all']=(Test['cc_cons_apr']+Test['cc_cons_may']+Test['cc_cons_jun'])/(Test['tot_spend_april']+Test['tot_spend_may']+Test['tot_spend_jun']+0.1)



# In[16]:


#creating new variable which is difference between amount debited for april to amount credited for april 
Data['diff_deb_crd_apr']=Data['debit_amount_apr']-Data['credit_amount_apr']
Data['diff_deb_crd_may']=Data['debit_amount_may']-Data['credit_amount_may']
Data['diff_deb_crd_jun']=Data['debit_amount_jun']-Data['credit_amount_jun']

#creating new variable which is difference between amount debited for april to amount credited for april 
Test['diff_deb_crd_apr']=Test['debit_amount_apr']-Test['credit_amount_apr']
Test['diff_deb_crd_may']=Test['debit_amount_may']-Test['credit_amount_may']
Test['diff_deb_crd_jun']=Test['debit_amount_jun']-Test['credit_amount_jun']


# In[17]:


#creating new variable which is total debit transaction in three months and total credit transaction in three months
Data['total_debit_inthreemonths']=Data['debit_amount_apr']+Data['debit_amount_may']+Data['debit_amount_jun']
Data['total_credit_inthreemonths']=Data['credit_amount_apr']+Data['credit_amount_may']+Data['credit_amount_jun']
Data['diff_oftotals']=((Data['debit_amount_apr']+Data['debit_amount_may']+Data['debit_amount_jun'])-(Data['credit_amount_apr']+Data['credit_amount_may']+Data['credit_amount_jun']))

#creating new variable which is
Test['total_debit_inthreemonths']=Test['debit_amount_apr']+Test['debit_amount_may']+Test['debit_amount_jun']
Test['total_credit_inthreemonths']=Test['credit_amount_apr']+Test['credit_amount_may']+Test['credit_amount_jun']
Test['diff_oftotals']=((Test['debit_amount_apr']+Test['debit_amount_may']+Test['debit_amount_jun'])-(Test['credit_amount_apr']+Test['credit_amount_may']+Test['credit_amount_jun']))


# In[18]:


#creating new variable which is total of all investments
Data['Total_investment']=Data['investment_1']+Data['investment_2']+Data['investment_3']+Data['investment_4']

Test['Total_investment']=Test['investment_1']+Test['investment_2']+Test['investment_3']+Test['investment_4']


# In[19]:


#From data file we are only taking the values which have credit card spend greater then 100 to prevent outliers
Data=Data[Data['cc_cons']>100].reset_index(drop=True)


# In[20]:


#Making y variable as credit card (Target) spend for next three months
y=Data['cc_cons']


# In[21]:


#Storing the values of id column from test file in ID
ID=Test['id']


# In[23]:


#Dropping the column credit card spend from Data file and storing it in x
x=Data.drop('cc_cons',axis=1)


# In[24]:


#Dropping the id columns in Data and Test files
X=x.drop('id',axis=1)
TEST=Test.drop('id',axis=1)


# In[25]:


#Replacing the values of Y and null in loan enquiry column with 0 and 1
X['loan_enq'] = np.where(x['loan_enq']=='Y', 1,0)
TEST['loan_enq']=np.where(Test['loan_enq']=='Y', 1,0)


# In[26]:


#Splitting the data from X file using train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[27]:


#Taking log1p of all values of train y 
Y_Train=np.log1p(y_train)


# In[28]:


#Declaring parameters for xgboost
param = {'max_depth':3,'eta':0.02,'silent':1,'subsample':0.6,'reg_lambda':1.5,'reg_alpha':0.001,
        'min_child_weight':7, 'colsample_bytree':0.85, 'nthread':32,'gamma':0.01,'objective':'reg:linear','tree_method':'approx',
        'booster':'gbtree'}


# In[29]:


#Training xgboost 
train_dmatrix=xgb.DMatrix(X_train,Y_Train)
model=xgb.train(param,train_dmatrix,num_boost_round = 800)
test_dmatrix=xgb.DMatrix(X_test)
TEST_dmatrix=xgb.DMatrix(TEST)


# In[30]:


#predicting and storing the values of splitted data using xgboost 
pred_test=model.predict(test_dmatrix)

#Converting the predicted values to normal using exponential as predicted values are in logarithmic form
prediction=np.expm1(pred_test)


# In[31]:


#Mean squared log error for splitted data is 
from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error( y_test, prediction ))


# In[32]:


#Mean absolute error for splitted data is 
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,prediction))


# In[33]:


#Now the prediction is done for test file for which the output values are not given
Pred_file=model.predict(TEST_dmatrix)
PREDICTION=np.expm1(Pred_file)
predic = pd.DataFrame(PREDICTION)
Final_prediction=pd.concat([ID.reset_index(drop=True),predic.reset_index(drop=True)],axis=1)


# In[35]:


#converting the dataframe to csv and storing it 
PREDCT =Final_prediction.to_csv (r'C:\Users\Final.csv', index = None)

