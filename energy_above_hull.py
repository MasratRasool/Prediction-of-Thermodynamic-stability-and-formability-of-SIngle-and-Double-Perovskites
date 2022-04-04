#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pymatgen.ext.matproj import MPRester
import pandas as pd
import pymatgen.analysis.phase_diagram as PhaseDiagram
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
API_KEY = "Bfq7dyNTipwl5vS2"


# In[3]:


with MPRester(API_KEY) as mpr:
    print(mpr.supported_properties)


# In[4]:


def make_query(query):
    with MPRester(API_KEY) as mpr:
      return mpr.get_entries(query,
        ['material_id',
        'pretty_formula',
        'icsd_id',
        'icsd_ids', 
        'e_above_hull',                            
        'spacegroup', 
        'energy_per_atom',
        'energy', 
        'composition'])
make_query("C")


# In[10]:


results = make_query("Fe-Bi-O")


# # Energy_above_Hull

# In[12]:


with MPRester(API_KEY) as mpr:
    query_results=mpr.query("Fe-Bi-O",
     ['material_id',
     'pretty_formula',
     'e_above_hull',                            
     ])
df = pd.DataFrame(query_results)
print(df)


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


# In[14]:


df = pd.read_csv("stability_database.csv")
df


# # Thermodynamic stability 

# In[15]:


sns.relplot(x="t", y="e_above_hull", hue="stable", data=df);


# In[16]:


df1 = pd.read_csv("stability dataset meV.csv")
df1


# In[17]:


sns.relplot(x="t", y="e_above_hull meV", hue="stable", style="stable",data=df1);


# In[18]:


df1.rename(columns = {'t':'Tolerance Factor (t)'}, inplace = True)
df1


# In[19]:


df2=df1
# convert A column is object type to category type
df2["A"] = df2["A"].astype('category')
df2["A"] = df2["A"].cat.codes
# convert A' column is object type to category type
df2["A'"] = df2["A'"].astype('category')
df2["A'"] = df2["A'"].cat.codes
# convert B column is object type to category type
df2["B"] = df2["B"].astype('category')
df2["B"] = df2["B"].cat.codes
# convert B' column is object type to category type
df2["B'"] = df2["B'"].astype('category')
df2["B'"] = df2["B'"].cat.codes
# convert X1 column is object type to category type
df2["X1"] = df2["X1"].astype('category')
df2["X1"] = df2["X1"].cat.codes
# convert type column is object type to category type
df2["type"] = df2["type"].astype('category')
df2["type"] = df2["type"].cat.codes
# convert functional group column is object type to category type
df2["functional group"] = df2["functional group"].astype('category')
df2["functional group"] = df2["functional group"].cat.codes
#df2
df1=df2


# In[20]:


df1


# # Loading Formablity database as df2

# In[39]:


df2 = pd.read_csv("formability_dataset.csv")


# In[40]:


df2.select_dtypes(include=['object'])


# In[41]:


# convert A column is object type to category type
df2["A"] = df2["A"].astype('category')
df2["A"] = df2["A"].cat.codes


# In[42]:


# convert A' column is object type to category type
df2["A'"] = df2["A'"].astype('category')
df2["A'"] = df2["A'"].cat.codes
# convert B column is object type to category type
df2["B"] = df2["B"].astype('category')
df2["B"] = df2["B"].cat.codes
# convert B' column is object type to category type
df2["B'"] = df2["B'"].astype('category')
df2["B'"] = df2["B'"].cat.codes
# convert X1 column is object type to category type
df2["X1"] = df2["X1"].astype('category')
df2["X1"] = df2["X1"].cat.codes
# convert type column is object type to category type
df2["type"] = df2["type"].astype('category')
df2["type"] = df2["type"].cat.codes
# convert functional group column is object type to category type
df2["functional group"] = df2["functional group"].astype('category')
df2["functional group"] = df2["functional group"].cat.codes
#df2


# In[110]:


df2.rename(columns = {'t':'Tolerance Factor (t)'}, inplace = True)
df2


# In[104]:


# import pandas as pd
# df2=df2.loc[df2["Perovskite"]==1]
# df2


# In[139]:


#df = df1.merge(df2, on='Tolerance Factor (t)')

df = df1.merge(df2, how='inner', left_on=["Tolerance Factor (t)","type","B_LUMO+"], right_on=["Tolerance Factor (t)","type","B_LUMO+"])
print (df)
df.describe()


# In[140]:


df


# # Energy_hull against tolerance factor for the intersection of two datasets

# In[142]:


sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
graph=sns.relplot(x="Tolerance Factor (t)", y="e_above_hull meV",style="Perovskite", hue="Perovskite",data=df);
plt.axhline(50,ls="--",c="green")

plt.axhline(200,ls="--",c="red")
plt.axhline(400,ls="--",c="red")
plt.legend(labels=["Ec=50meV"])


plt.ylim(0, 800)
plt.show()


# In[ ]:




