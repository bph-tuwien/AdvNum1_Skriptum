#!/usr/bin/env python
# coding: utf-8

# # Implementation - Some general thoughts
# A brief overview of what to look out for when implementing numerical concepts and some personal preferences will be given.
# 
# ```{important}
# It is good to make your mind up about this stuff before you start to code, but don't waste your time too long on this. As long as you are not working on insanely huge projects, you can always optimize and restructure your code later. The first step for the numerical analyst is to produce results, but after that you may consider to clean up the potential mess you created.
# ```
# 
# ## What to look out for
# ### Unnecessary looping
# This works:

# ```python
# for i in range(K_Array.shape[0]):
#         for j in range(K_Array.shape[1]):
#             index = str(i+1) + str(j+1)
#             for K in K_List:
#                 if K.index == index:
#                     K_Array[i, j] = K.K_value
#                     K_Array[j, i] = K.K_value
# ```
# 

# But we have a lot of loops we have to iterate through. When using Numpys `ndenumerate` function we can easily get rid of one of the loops.

# ```python
#     for index in np.ndindex(K_Array):
#         i = index[0]    # Row index
#         j = index[1]    # Column index
#         str_index = str(i+1) + str(j+1)
#         for K in K_List:
#             if K.index == str_index:
#                 K_Array[i, j] = K.K_value
#                 K_Array[j, i] = K.K_value
# ```

# ```{note}
# Think of each loop and ask yourself if you are really needing it. Are there built in functions which can do the same? They will mostly be faster than whatever you can implement in a feasible amount of time. Always try to vecotrize before doing anything else!
# ```

# ### Smarter ways of implementation
# 
# Let's look again at this:
# 
# ```python
# for index, elem in np.ndenumerate(K_Array):
#     i = index[0]    # Row index
#     j = index[1]    # Column index
#     str_index = str(i+1) + str(j+1)
#     for K in K_List:
#                 if K.index == str_index:
#                     K_Array[i, j] = K.K_value
#                     K_Array[j, i] = K.K_value
# ```
# 
# It seems like we are trying to find some identifier which happens to be equal to some position in the matrix.
# This UID seems to be stored as a parameter of the object ```K```. Why not create a dictionary where we can directly access these elements based on their unique identifier?
# 
# ```python
# 
# K_index_list = [K.index for K in K_List]
# K_value_list = [K.K_value for K in K_List]
# K_value_dict = dict(zip(K_index_list, K_value_list))
# 
# for index, elem in np.ndenumerate(K_Array):
#     i = index[0]    # Row index
#     j = index[1]    # Column index
#     str_index = str(i+1) + str(j+1)
#     if str_index in K_value_dict.keys():
#         K_Array[i, j] = K_value_dict[str_index]
#         K_Array[j, i] = K_value_dict[str_index]     # Symmetrical Matrix
# ```
# 
# ### Object oriented programming
# Especially for prototyping where readability and ease of implementation are of interest, object oriented programming (OOP) may help you code concepts and ideas in a more straightforward way. Furthermore, when projects get bigger, it is easier to continue to work with the same code and expand it when needed.
# 
# For example, if I have problem where I want to calculate the U-value of a wall of course I can do it like this:

# In[1]:


# Calculate U-value#

def u_value(Rse, thickness, conduction, Rsi):
    '''
    Calculates U-value
    '''

    return 1 / (Rse + thickness / conduction + Rsi)

# First wall
conduction_1 = 2.3 # in W/mK
thickness_1 = 0.2 # in meters
Rsi_1 = 0.13 # in m²K/W
Rse_1 = 0.04 # in m²K/W

# Second wall
conduction_2 = 1.2 # in W/mK
thickness_2 = 0.2 # in meters
Rsi_2 = 0.13 # in m²K/W
Rse_2 = 0.04 # in m²K/W

u_wall_1 = u_value(Rse_1, thickness_1, conduction_1, Rsi_1)
u_wall_2 = u_value(Rse_2, thickness_2, conduction_2, Rsi_2)

(u_wall_1, u_wall_2)


# But with almost the same effort I could have created a `Class Wall` which can have the needed parameters and the function to calculate its U-value.

# In[2]:


class Wall(object):
    def __init__(self, Rse, thickness, conduction, Rsi):
        self.thickness = thickness
        self.conduction = conduction
        self.Rse = Rse
        self.Rsi = Rsi

    def u_value(self):
        return 1 / (self.Rse + self.thickness / self.conduction + self.Rsi)


wall1 = Wall(Rse=0.04, thickness=0.2, conduction=2.3, Rsi=0.13)
wall2 = Wall(Rse=0.04, thickness=0.2, conduction=1.2, Rsi=0.13)

(wall1.u_value(), wall2.u_value())


# And if I see while I am working on the project that I need further information to define my wall or it needs to have more functions for certain evaluations, I can simply add those as parameters to my main class.

# In[3]:


class Wall(object):
    def __init__(self, Rse, thickness, conduction, Rsi, some_extra_parameter):
        self.thickness = thickness
        self.conduction = conduction
        self.Rse = Rse
        self.Rsi = Rsi
        self.some_extra_parameter = some_extra_parameter

    def u_value(self):
        return 1 / (self.Rse + self.thickness / self.conduction + self.Rsi)

    def super_mysterious_optimized_calculation(self):
        return # something super_mysterious_optimized

