
from scipy.optimize import leastsq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r'../data/04_cricket_1999to2011.csv')


# # Data Cleanup

# In[15]:


df_new = df[df['Innings']==1]
df_new = df_new[df_new['Error.In.Data']==0]
df_new['Over'] = df['Total.Overs'] - df_new['Over']
df_new = df_new[['Over','Wickets.in.Hand','Runs.Remaining']]
df_new.columns = ['Over.Remaining','Wickets.in.Hand','Runs.Remaining']
new_data = df_new[df_new['Wickets.in.Hand']!=0]


# In[16]:


Wickets_rem = list(new_data['Wickets.in.Hand'])
Overs_rem = list(new_data['Over.Remaining'])
Runs_rem = list(new_data['Runs.Remaining'])


# # Error Function

# In[17]:


x0 = [10,20,30,40,50,60,70,80,90,100,10] #initial parameters

def error_func(z0,w,u,z):
    net_loss = 0
    z_pred = []
    z = np.array(z)
    L = z0[10]
    for i in range(len(w)):
        z_pred.append(z0[w[i]-1]*(1-np.exp(-1*L*u[i]/z0[w[i]-1])))
    z_pred = np.array(z_pred)
    return z_pred-z


# In[18]:


model = leastsq(error_func,x0,args=(Wickets_rem,Overs_rem,Runs_rem)) #model least squares


# In[28]:


o = np.linspace(0,50,301)
L = model[0][10]
opt_para = model[0][0:10]
plt.figure(figsize=(6,4),dpi=150)

for i in opt_para:
    z = i*(1-np.exp(-L*o/i))
    plt.plot(50-o,z)
plt.xlabel('Overs Used')
plt.ylabel('Average Runs Possible')
temp = ['Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10']
plt.legend(temp)
plt.show()

# # Total loss

# In[22]:


t = error_func(model[0],Wickets_rem,Overs_rem,Runs_rem)
np.sum(t**2)/len(Wickets_rem)


# In[ ]:

print('Mean Squared Error:',mse)
print('Optimized Z1 t0 Z10 :',opt_para)


