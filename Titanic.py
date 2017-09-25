
# coding: utf-8

# # 一、数据概览(Data Abstract)

# In[1]:


import pandas as pd
titanic_df = pd.read_csv('titanic-data.csv')
print "Titanic数据前五行："
titanic_df.head()


# In[2]:


print "Titanic各列的统计数据："
titanic_df.describe()


# # 二、提出问题(Question Phase)

# 我提出的问题是：Titanic事故中，生存率与哪些变量有关系，我希望考虑的变量有：船舱等级(Pclass)、性别(Sex)、年龄(Age)、同行亲属数量(SibSp)和票价(Fare)

# # 三、数据整理(Wrangling Phase)

# 首先，我们检查数据的完整性

# In[3]:


titanic_df.info()


# 可以看出，该数据集总共有891行观测值，他们的索引为从0至890，数据集包含12个列变量，其中列变量Age、Cabin和Embarked存在缺失值。
# 

# 其中年龄(Age)缺失值不算很多，但是年龄(Age)是我们分析的重要变量，因此需要处理Age缺失的行数据，这里我将Age缺失的行数据删除掉。

# 舱位号(Cabin)的缺失值非常多，但它不是我们分析的变量，因此对该缺失值不做处理。

# 列变量是否登船(Embarked)仅有两个缺失值，但是若这两位乘客未登船，他们将不会在Titanic事故现场，这会影响我们对生存率的分析，因此我将这两行数据删除掉。

# 下一行代码我对数据进行了以上处理。

# In[4]:


titanic_df.dropna(axis = 0, subset = ['Age', 'Embarked'], inplace = True)


# 再次检查数据的完整性

# In[5]:


titanic_df.info()


# 可以看出，经过数据筛选，当前该数据集总共有712行观测值，除了舱位号(Cabin)外，其他列均没有缺失值。

# # 四、探索阶段(Explore Phase)

# ## 1、单变量分析

# ### (1) 整体(Abstract)

# In[6]:


titanic_df.describe()


# Titanic的乘客平均年龄为29.64岁，有记录的年龄最小乘客为0.42岁，年龄最大乘客为80岁。
# 

# 在有年龄记录的乘客中，生存率为40.45%。

# 船舱等级分为3个等级，分别是1级、2级和3级，乘客的平均船舱等级为2.24。

# 平均每名乘客有0.51位同行亲属，有些乘客单独出行，没有同行亲属，有些乘客拥有最高的5位同行亲属。

# 乘客中的平均票价是34.57，最便宜的票价是0，最贵的票价是512.33。

# ### (2) 乘客年龄(Age)

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
titanic_age_df = titanic_df[['Age']]
plt.hist(np.array(titanic_age_df), bins = 40, range = [0, 80])
plt.title('The age distribution of passengers')
plt.xlabel("Age")
plt.ylabel("The number of passengers")


# Titanic的乘客大多为16至40岁，其中18至32岁的青壮年最多。乘客中，10岁以下儿童较多，10岁至18岁的青少年较少。60岁以上的老年人较少。

# ### (3) 船舱等级(Pclass)

# In[8]:


import matplotlib.mlab as mlab

titanic_pclass_df = titanic_df[['Pclass']]
titanic_pclass_list = [0, 0, 0]
for index in titanic_pclass_df.iterrows():
    if index[1][0] == 1:
        titanic_pclass_list[0] += 1
    if index[1][0] == 2:
        titanic_pclass_list[1] += 1
    if index[1][0] == 3:
        titanic_pclass_list[2] += 1
pclass_x = [1, 2, 3]  
pclass_y = [titanic_pclass_list]
print "1、2、3等级的船舱的人数分别为："
print pclass_y
plt.bar(pclass_x, titanic_pclass_list, 0.4)
plt.title("The number of passengers with various Pclass")
plt.xlabel("Pclass")
plt.ylabel("The number of passengers")
plt.show()    


# 1级船舱的乘客有184人，2级船舱的乘客有173人，3级船舱的乘客有355人。可以看出，3级船舱乘客最多，1级船舱比2级船舱乘客数量略多。

# ### (4) 性别(Sex)

# In[9]:


titanic_sex_df = titanic_df[['Sex']]
titanic_sex_list = [0, 0]
for index in titanic_sex_df.iterrows():
    if index[1][0] == 'male':
        titanic_sex_list[0] += 1
    if index[1][0] == 'female':
        titanic_sex_list[1] += 1
labels = ['male passengers: ' + str(titanic_sex_list[0]) + ' | ' + str((titanic_sex_list[0]*1.0 / len(titanic_sex_df)*1.0)*100.0)[: 5] + '%',          'female passengers: '  + str(titanic_sex_list[1]) + ' | ' + str((titanic_sex_list[1]*1.0 / len(titanic_sex_df)*1.0)*100.0)[: 5] + "%"]
sizes = [titanic_sex_list[0], titanic_sex_list[1]]
colors = ['lightblue', 'pink']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.show()


# 男性乘客有453人，女性乘客有259人，男性乘客占比63.62%，女性乘客占比36.37%，男性乘客比女性乘客更多。

# ### (5) 同行亲属数量(SibSp)

# In[10]:


titanic_sibsp_df = titanic_df[['SibSp']]
titanic_sibsp_list = [0, 0, 0, 0, 0, 0]
for index in titanic_sibsp_df.iterrows():
    if index[1][0] == 0:
        titanic_sibsp_list[0] += 1
    if index[1][0] == 1:
        titanic_sibsp_list[1] += 1
    if index[1][0] == 2:
        titanic_sibsp_list[2] += 1
    if index[1][0] == 3:
        titanic_sibsp_list[3] += 1
    if index[1][0] == 4:
        titanic_sibsp_list[4] += 1
    if index[1][0] == 5:
        titanic_sibsp_list[5] += 1
sibsp_x = [0, 1, 2, 3, 4, 5]  
sibsp_y = [titanic_sibsp_list]
plt.bar(sibsp_x, titanic_sibsp_list, 0.4)
plt.title("The number of passengers with various SibSp")
plt.xlabel("SibSp")
plt.ylabel("The number of passengers")
plt.show() 

labels = ['0 SibSp: ' + str(titanic_sibsp_list[0]) + ' | ' + str((titanic_sibsp_list[0]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%',          '1 SibSp: ' + str(titanic_sibsp_list[1]) + ' | ' + str((titanic_sibsp_list[1]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%',          '2 SibSp: ' + str(titanic_sibsp_list[2]) + ' | ' + str((titanic_sibsp_list[2]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%',          '3 SibSp: ' + str(titanic_sibsp_list[3]) + ' | ' + str((titanic_sibsp_list[3]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%',          '4 SibSp: ' + str(titanic_sibsp_list[4]) + ' | ' + str((titanic_sibsp_list[4]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%',          '5 SibSp: ' + str(titanic_sibsp_list[5]) + ' | ' + str((titanic_sibsp_list[5]*1.0 / len(titanic_sibsp_df)*1.0)*100.0)[: 5] + '%']
sizes = [titanic_sibsp_list[0], titanic_sibsp_list[1], titanic_sibsp_list[2], titanic_sibsp_list[3], titanic_sibsp_list[4], titanic_sibsp_list[5]]
colors = ['pink', 'lightyellow', 'lightblue', 'lightgreen', 'lightcyan', 'violet']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.show()


# 可以看出近2/3的乘客都是独自出行，约1/4的乘客有一位同行亲属，少量的乘客有两人或两人以上的同行亲属。

# ### (6) 票价(Fare)

# In[11]:


titanic_fare_df = titanic_df[['Fare']]
plt.hist(np.array(titanic_fare_df), bins = 57, range = [0, 513])
plt.title('The fare distribution of passengers')
plt.xlabel("fare")
plt.ylabel("The number of passengers")


# 

# 这里，我们发现一个异常情况，大多乘客的票价在0至280之间，没有280至500价位的船票，但有极少量的乘客的票价在500多，我猜测那是类似于总统套房一样的奢侈舱位。这个过大的值对我们的分析造成了干扰，所以我再绘制一幅过滤了这些极大值的直方图。

# In[12]:


titanic_fare_dropped_df = titanic_fare_df[(titanic_fare_df.Fare < 500)]
plt.hist(np.array(titanic_fare_dropped_df), bins = 40, range = [0, 280])
plt.title('The fare distribution of passengers')
plt.xlabel("fare")
plt.ylabel("The number of passengers")


# 可以看出，除去极少数票价高于500的奢侈舱位，大多的票价都集中在0至40之间，这些应该属于大量的廉价舱位。

# 中等舱位的票价大多集中在50至90之间，另有少量110至160之间和210至270之间的高等舱位。

# 票价的购买人数分布大体遵循帕累托定律(Pareto Principle)，廉价舱位人数非常多，而高等舱位和奢侈舱位人数极少。

# ## 2、多变量分析

# ### (1) 生存率与乘客年龄的联系(Survived - Age)

# 因为Titanic的乘客年龄分布不均，所以需要分析各年龄段的具体生存和死亡状况，以下代码我构建了不同年龄段乘客生存和死亡的柱形图和堆积图。

# In[13]:


titanic_sur_age_df = titanic_df[['Survived', 'Age']]
titanic_age_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0] #各年龄段的人数
titanic_sur_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0] #各年龄段的生存者人数
titanic_dead_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0] #各年龄段的死亡人数
titanic_sur_rate_list = [0, 0, 0, 0, 0, 0, 0, 0, 0] #各年龄段的生存率

#构造titanic_age_count_list、titanic_sur_count_list、titanic_dead_count_list和titanic_sur_rate_list四个列表
i = 0
while i < len(titanic_age_count_list):
    titanic_age_count_list[i] = len(titanic_sur_age_df[(titanic_sur_age_df.Age >= i * 10.0) & (titanic_sur_age_df.Age < (i + 1) * 10.0)])
    titanic_sur_count_list[i] = len(titanic_sur_age_df[(titanic_sur_age_df.Age >= i * 10.0) & (titanic_sur_age_df.Age < (i + 1) * 10.0) &                                               (titanic_sur_age_df.Survived == 1.0)])
    titanic_dead_count_list[i] = titanic_age_count_list[i] - titanic_sur_count_list[i]
    titanic_sur_rate_list[i] = (titanic_sur_count_list[i] * 1.0) / (titanic_age_count_list[i] * 1.0)
    i = i + 1

#构造titanic_sur_dead_count_list列表
titanic_sur_dead_count_list = []
j = 0
while j < len(titanic_sur_count_list):
    titanic_sur_dead_count_list.append([titanic_sur_count_list[j], titanic_dead_count_list[j]])
    j = j + 1

#构造titanic_sur_dead_count_df
titanic_sur_dead_count_df = pd.DataFrame(                                      data = titanic_sur_dead_count_list,                                      index = ['0 - 10','10 - 20','20 - 30','30 - 40','40 - 50', '50 - 60','60 - 70','70 - 80','80 - 90'],                                      columns = ['survived','dead'])

print "各年龄段的生存者和死亡者人数："
print titanic_sur_dead_count_df

#生成柱状图和堆积图
titanic_sur_dead_count_df.plot(kind = 'bar', title = 'The number of passengers survived and dead at all ages')
titanic_sur_dead_count_df.plot(kind = 'barh', stacked = True, title = 'The number of passengers survived and dead at all ages')


# 从这两幅图表来看，各年龄段的生还者普遍少于死亡者，可以看出Titanic各年龄段的生存率普遍偏低。

# 各年龄段的生存和死亡人数大体成正比，生存人数和该年龄段总人数保持一定比例，使得人数较多的年龄段的乘客生还者较多，人数较少的年龄段的乘客生还者较少。

# 不过我们似乎可以观察到一个细节，低于10岁的儿童和高于80岁的老人是唯一两个生还者多于死亡者的群体，而20岁至30岁的年轻人死亡人数看起来远远大于幸存者人数，这种悬殊相比其他年龄段更大。

# 为此，在下一段代码我绘制了个年龄段生存率的柱形图，进行进一步分析。

# In[14]:


titanic_sur_rate_df = pd.DataFrame(                                data = titanic_sur_rate_list,                                index = ['0 - 10','10 - 20','20 - 30','30 - 40','40 - 50', '50 - 60','60 - 70','70 - 80','80 - 90'],                                columns = ['survived rate'])
titanic_sur_rate_df.plot(kind = 'bar', title = 'survived rate at all ages')


# 可以看到，10岁至70岁各个年龄段的生存率大体持平，这印证了前面的分析，生存人数和该年龄段总人数保持一定比例，各年龄段的生存和死亡人数大体成正比。但是，我们也有以下发现：

# 0至10岁年龄段的乘客有较高的生存率，我猜测这是因为Titanic的船员在紧急情况下将救生艇更多地留给了儿童，也体现了乘客在面对生命的抉择时往往会舍弃自己而为孩子留下更多生存的机会。

# 60至70岁和20至30岁两个年龄段的生存率相对更低，这印证了前面的分析，20岁至30岁的生存和死亡比例悬殊相对更大(60至70岁年龄段因为人数较少，在柱形图中不易察觉)，我猜测这是因为处于这两个年龄段的乘客大多是0至10岁那部分儿童乘客的父母或祖父母，他们可能在紧急时刻放弃了自己的生命，把机会留给了孩子，这也印证了我在上一项分析中的猜测。

# 70至80岁的生存率为0，结合前面的图表可知，这个年龄段有6个人，全部未能幸存，或许这只是一种巧合，或许像我们在《Titanic》中所看到的那样，这个年龄的人或许更能够冷静地直面死亡。

# 80至90岁的生存率为100%，这引起了我的好奇，不过观察前面的图表可知，有记录的这个年龄段的乘客只有1人，而这1人恰好幸存了下来，所以造成了这个令人惊讶的生存率，这只是一种巧合。

# 不过，以上的猜测不一定成立，还有更多的干扰因素可能导致这样不同的生存率，在紧急状况下，乘客所在的位置，求生的能力、船员的营救状况以及一些突发事件等等因素，都会影响生存率，所以以上猜测并不能说明生存率与年龄之间存在必然的联系。

# ### (2) 生存率与船舱等级的联系(Survived - Pclass)

# 三个等级的船舱人数不同，其中3级船舱人数较多，1级和2级船舱人数相近，但三个等级的船舱都是有一百多或三百多人，数量较多，可以直接比较三个等级船舱的生存率，以下代码我绘制了三个等级船舱的生存人数柱状图、堆积图和生存率的柱形图。

# In[15]:


titanic_sur_pclass_df = titanic_df[['Survived', 'Pclass']]
titanic_pclass_count_list = [0, 0, 0] #各船舱等级的人数
titanic_sur_pclass_count_list = [0, 0, 0] #各船舱等级的生存者人数
titanic_dead_pclass_count_list = [0, 0, 0] #各船舱等级的死亡人数
titanic_sur_pcalss_rate_list = [0, 0, 0] #各船舱等级的生存率

#将titanic_sur_pclass_df转换为list，方便遍历统计
titanic_sur_pclass_np = np.array(titanic_sur_pclass_df)
titanic_sur_pclass_list = titanic_sur_pclass_np.tolist()

#构造titanic_pclass_count_list、titanic_sur_pclass_count_list、titanic_sur_pcalss_rate_list三个列表
i = 0
while i < len(titanic_pclass_count_list):
    j = 0
    count_pclass = 0
    while j < len(titanic_sur_pclass_list):
        if int(titanic_sur_pclass_list[j][1]) == i + 1:
            count_pclass = count_pclass + 1
        j = j + 1
    titanic_pclass_count_list[i] = count_pclass
    
    j = 0
    count_sur_pclass = 0
    while j < len(titanic_sur_pclass_list):
        if int(titanic_sur_pclass_list[j][1]) == i + 1 and int(titanic_sur_pclass_list[j][0]) == 1:
            count_sur_pclass = count_sur_pclass + 1
        j = j + 1
    titanic_sur_pclass_count_list[i] = count_sur_pclass

    titanic_dead_pclass_count_list[i] = titanic_pclass_count_list[i] - titanic_sur_pclass_count_list[i]
    titanic_sur_pcalss_rate_list[i] = (titanic_sur_pclass_count_list[i] * 1.0) / (titanic_pclass_count_list[i] * 1.0)
    i = i + 1

#构造titanic_sur_dead_pclass_count_list列表
titanic_sur_dead_pclass_count_list = []
j = 0
while j < len(titanic_pclass_count_list):
    titanic_sur_dead_pclass_count_list.append([titanic_sur_pclass_count_list[j], titanic_dead_pclass_count_list[j]])
    j = j + 1

#构造titanic_sur_dead_count_df
titanic_sur_dead_pclass_count_df = pd.DataFrame(                                      data = titanic_sur_dead_pclass_count_list,                                      index = ['Pclass_1', 'Pclass_2', 'Pclass_3'],                                      columns = ['survived','dead'])

print "各舱位的生存者和死亡者人数："
print titanic_sur_dead_pclass_count_df

#生成柱状图和堆积图
titanic_sur_dead_pclass_count_df.plot(kind = 'bar', title = 'The number of passengers survived and dead at all pclasses')
titanic_sur_dead_pclass_count_df.plot(kind = 'barh', stacked = True, title = 'The number of passengers survived and dead at all pclasses')

#生成各等级舱位生存率的柱状图
titanic_sur_pcalss_rate_df = pd.DataFrame(                                data = titanic_sur_pcalss_rate_list,                                index = ['Pclass_1', 'Pclass_2', 'Pclass_3'],                                columns = ['survived rate'])
titanic_sur_pcalss_rate_df.plot(kind = 'bar', title = 'survived rate at every pclass')


# 从前两幅图来看，3个等级的舱位的乘客的生存状况有很大差异，1级舱位有较多的乘客生存下来；2级舱位幸存和死亡的乘客相对持平，死亡的乘客略多于幸存的乘客；3级舱位死亡的乘客远多于幸存者。

# 通过第三幅图可以看出，1级舱位、2级舱位、3级舱位的生存率逐个递减。3级舱位死亡数量巨大，不仅仅是因为3级舱位乘客数量基数很大，更是因为3级舱位本身生存率就很低。

# 通过数据分析，我猜测更高级的舱位的乘客享有了优先逃生的机会，这可能是因为高级舱位提供了足够的逃生设备，或者高级舱位所处的位置更加安全，也有可能船员救生时为高级舱位的乘客提供了“绿色通道”。

# 但以上猜测也不一定是绝对成立的，其他的干扰因素可能也会导致这样的生存率差异，例如3级舱位可能因为人数众多对信息的获取更慢，导致逃生的时间更少。

# ### (3) 生存率与性别的联系(Survived - Sex)

# 以下代码我绘制了不同性别的生存人数柱状图、堆积图和生存率的柱形图。

# In[16]:


titanic_sur_sex_df = titanic_df[['Survived', 'Sex']]
titanic_sex_count_list = [0, 0] #不同性别的人数
titanic_sur_sex_count_list = [0, 0] #不同性别的生存者人数
titanic_dead_sex_count_list = [0, 0] #不同性别的死亡人数
titanic_sur_rate_sex_list = [0, 0] #不同性别的生存率

#titanic_sex_count_list、titanic_sur_count_list、titanic_dead_count_list和titanic_sur_rate_list四个列表
titanic_sex_count_list[0] = len(titanic_sur_sex_df[(titanic_sur_sex_df.Sex == 'male')])
titanic_sex_count_list[1] = len(titanic_sur_sex_df[(titanic_sur_sex_df.Sex == 'female')])
titanic_sur_sex_count_list[0] = len(titanic_sur_age_df[(titanic_sur_sex_df.Sex == 'male') & (titanic_sur_age_df.Survived == 1.0)])
titanic_sur_sex_count_list[1] = len(titanic_sur_age_df[(titanic_sur_sex_df.Sex == 'female') & (titanic_sur_age_df.Survived == 1.0)])
titanic_dead_sex_count_list[0] = len(titanic_sur_age_df[(titanic_sur_sex_df.Sex == 'male') & (titanic_sur_age_df.Survived == 0.0)])
titanic_dead_sex_count_list[1] = len(titanic_sur_age_df[(titanic_sur_sex_df.Sex == 'female') & (titanic_sur_age_df.Survived == 0.0)])
titanic_sur_rate_sex_list[0] = (titanic_sur_sex_count_list[0] * 1.0) / (titanic_sex_count_list[0] * 1.0)
titanic_sur_rate_sex_list[1] = (titanic_sur_sex_count_list[1] * 1.0) / (titanic_sex_count_list[1] * 1.0)

#构造titanic_sur_dead_sex_count_list列表
titanic_sur_dead_sex_count_list = []
j = 0
while j < len(titanic_sex_count_list):
    titanic_sur_dead_sex_count_list.append([titanic_sur_sex_count_list[j], titanic_dead_sex_count_list[j]])
    j = j + 1

#构造titanic_sur_dead_count_df
titanic_sur_dead_sex_count_df = pd.DataFrame(                                      data = titanic_sur_dead_sex_count_list,                                      index = ['male', 'female'],                                      columns = ['survived','dead'])

print "不同性别的生存者和死亡者人数："
print titanic_sur_dead_sex_count_df

#生成柱状图和堆积图
titanic_sur_dead_sex_count_df.plot(kind = 'bar', title = 'The number of passengers survived and dead at each gender')
titanic_sur_dead_sex_count_df.plot(kind = 'barh', stacked = True, title = 'The number of passengers survived and dead at each gender')

#生成不同性别生存率的柱状图
titanic_sur_rate_sex_df = pd.DataFrame(                                data = titanic_sur_rate_sex_list,                                index = ['male', 'female'],                                columns = ['survived rate'])
titanic_sur_rate_sex_df.plot(kind = 'bar', title = 'survived rate at each gender')


# 在之前对乘客性别的单变量分析中，我们获知男性乘客多余乘客女性。然而，通过本多变量分析可以看出，男性乘客的幸存者却远远少于女性，男性乘客的死亡人数也远远大于女性乘客。

# 通过第三幅图可以看出，男性乘客的生存率仅有20%左右，女性乘客的生存率高达75%左右。男性乘客的生存率远远低于女性乘客，差距非常悬殊。

# 我猜测造成该生存率的巨大悬殊可能是因为船员救生的时候更多地照顾了女性乘客，也有可能在灾难发生时， 大多男性乘客把生存的机会留给了自己的爱人(类似于《Titanic》的剧情，Jack把生存的机会留给了Rose)。

# 但以上也仅仅是一种猜测，我们不能断定生存率必然和性别存在联系。其他的干扰因素可能依然存在，例如，女性乘客的客舱位置可能相对男性更加安全，或者女性乘客的年龄普遍更加年轻，具备更强的求生能力。这些都会造成该生存率在性别上的差异。

# ### (4) 生存率与同行亲属数量的联系(Survived - SibSp)

# 不同同行亲属数量的人数差异较大，近2/3的乘客都是独自出行，约1/4的乘客有一位同行亲属，少量的乘客有两人或两人以上的同行亲属，所以我们需要使用柱状图和堆积图来详细分析不同亲属数量的情况。以下代码我绘制了不同亲属数量的生存人数柱状图、堆积图和生存率的柱形图。

# In[17]:


titanic_sur_SibSp_df = titanic_df[['Survived', 'SibSp']]
titanic_SibSp_count_list = [0, 0, 0, 0, 0, 0,] #不同同行亲属的人数
titanic_sur_SibSp_count_list = [0, 0, 0, 0, 0, 0] #不同同行亲属的生存者人数
titanic_dead_SibSp_count_list = [0, 0, 0, 0, 0, 0] #不同同行亲属的死亡人数
titanic_sur_SibSp_rate_list = [0, 0, 0, 0, 0, 0] #不同同行亲属的生存率

#构造titanic_age_count_list、titanic_sur_count_list、titanic_dead_count_list和titanic_sur_rate_list四个列表
i = 0
while i < len(titanic_SibSp_count_list):
    titanic_SibSp_count_list[i] = len(titanic_sur_SibSp_df[(titanic_sur_SibSp_df.SibSp == i)])
    titanic_sur_SibSp_count_list[i] = len(titanic_sur_SibSp_df[(titanic_sur_SibSp_df.SibSp == i) & (titanic_sur_SibSp_df.Survived == 1.0)])
    titanic_dead_SibSp_count_list[i] = len(titanic_sur_SibSp_df[(titanic_sur_SibSp_df.SibSp == i) & (titanic_sur_SibSp_df.Survived == 0.0)])
    titanic_sur_SibSp_rate_list[i] = (titanic_sur_SibSp_count_list[i] * 1.0) / (titanic_SibSp_count_list[i] * 1.0)
    i = i + 1

#构造titanic_sur_dead_count_list列表
titanic_sur_dead_SibSp_count_list = []
j = 0
while j < len(titanic_SibSp_count_list):
    titanic_sur_dead_SibSp_count_list.append([titanic_sur_SibSp_count_list[j], titanic_dead_SibSp_count_list[j]])
    j = j + 1

#构造titanic_sur_dead_count_df
titanic_sur_dead_SibSp_count_df = pd.DataFrame(                                      data = titanic_sur_dead_SibSp_count_list,                                      index = ['0', '1', '2', '3', '4', '5'],                                      columns = ['survived','dead'])

print "不同亲属数量的生存者和死亡者人数："
print titanic_sur_dead_SibSp_count_df

#生成柱状图和堆积图
titanic_sur_dead_SibSp_count_df.plot(kind = 'bar', title = 'The number of passengers survived and dead at every SibSp')
titanic_sur_dead_SibSp_count_df.plot(kind = 'barh', stacked = True, title = 'The number of passengers survived and dead at every SibSp')

#生成不同亲属数量的乘客生存率柱状图
titanic_sur_dead_SibSp_count_df = pd.DataFrame(                                data = titanic_sur_SibSp_rate_list,                                index = ['0', '1', '2', '3', '4', '5'],                                columns = ['survived rate'])
titanic_sur_dead_SibSp_count_df.plot(kind = 'bar', title = 'survived rate at every SibSp')


# 大多数乘客是单独出行或携带了一位亲属，携带两个或两个以上亲属的乘客数量较少，通过前两幅图观察他们幸存和死亡的人数，可以看出单独出行的乘客死亡的人数最多，但幸存的人数也最多，这是因为单独出行的乘客占了大多数。通过对比发现，携带一位亲属的乘客是唯一幸存者多于死亡人数的群体。

# 通过观察第三幅图，除了单独出行的乘客，其他携带1位亲属至5位亲属的乘客生存率逐个降低。我推测这是因为单独出行的乘客需要独自完成自救，求生能力不高，而携带一位亲属的乘客在面对自己的一个家人时具有更强的求生欲望和求生能力，他们可以协作完成逃生。携带过多亲属的乘客会因为需要照顾太多的亲人而拖延求生时间，导致了他们生存率更低。

# 以上猜测也不一定能够证明生存率与乘客携带亲属数量存在必然的联系，依然存在一些干扰因素。在这个数据集中，携带两个或两个以上亲属的乘客数量太少，该生存率的数据可能不具有代表性。

# ### (5) 生存率与票价的联系(Survived - Fare) 

# 因为票价遵循了帕累托定律(Pareto Principle)，各个票价的乘客数量非常不均衡，因此我们同样采用柱形图和累计图结合生存率柱形图进行分析。以下代码我绘制了不同票价乘客的生存人数柱状图、堆积图和生存率的柱形图。

# In[18]:


titanic_sur_fare_df = titanic_df[['Survived', 'Fare']]
titanic_fare_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #各票价区间的人数
titanic_sur_fare_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #各票价区间的生存者人数
titanic_dead_fare_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #各票价区间的死亡人数
titanic_sur_fare_rate_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #各票价区间的生存率

#构造titanic_age_count_list、titanic_sur_count_list、titanic_dead_count_list和titanic_sur_rate_list四个列表
i = 0
while i < len(titanic_fare_count_list):
    titanic_fare_count_list[i] = len(titanic_sur_fare_df[(titanic_sur_fare_df.Fare >= i * 50.0) & (titanic_sur_fare_df.Fare < (i + 1) * 50.0)])
    titanic_sur_fare_count_list[i] = len(titanic_sur_fare_df[(titanic_sur_fare_df.Fare >= i * 50.0) & (titanic_sur_fare_df.Fare < (i + 1) * 50.0) &                                                            (titanic_sur_fare_df.Survived == 1.0)])
    titanic_dead_fare_count_list[i] = titanic_fare_count_list[i] - titanic_sur_fare_count_list[i]
    if titanic_fare_count_list[i] != 0:
        titanic_sur_fare_rate_list[i] = (titanic_sur_fare_count_list[i] * 1.0) / (titanic_fare_count_list[i] * 1.0)
    else:
        titanic_sur_fare_rate_list[i] = 0
    i = i + 1

#构造titanic_sur_dead_count_list列表
titanic_sur_dead_fare_count_list = []
j = 0
while j < len(titanic_sur_fare_count_list):
    titanic_sur_dead_fare_count_list.append([titanic_sur_fare_count_list[j], titanic_dead_fare_count_list[j]])
    j = j + 1

#构造titanic_sur_dead_count_df
titanic_sur_dead_fare_count_df = pd.DataFrame(                                      data = titanic_sur_dead_fare_count_list,                                      index = ['0 - 50','50 - 100','100 - 150','150 - 200','200 - 250', '250 - 300', '300 - 350',                                              '350 - 400','400 - 450','450 - 500','500 - 550','550 - 600'],                                      columns = ['survived','dead'])

print "各票价区间的生存者和死亡者人数："
print titanic_sur_dead_fare_count_df

#生成柱状图和堆积图
titanic_sur_dead_fare_count_df.plot(kind = 'bar', title = 'The number of passengers survived and dead at all fares')
titanic_sur_dead_fare_count_df.plot(kind = 'barh', stacked = True, title = 'The number of passengers survived and dead at all fares')

titanic_sur_fare_rate_df = pd.DataFrame(                                data = titanic_sur_fare_rate_list,                                index = ['0 - 50','50 - 100','100 - 150','150 - 200','200 - 250', '250 - 300', '300 - 350',                                         '350 - 400','400 - 450','450 - 500','500 - 550','550 - 600'],                                columns = ['survived rate'])
titanic_sur_fare_rate_df.plot(kind = 'bar', title = 'survived rate at all fares')


# 通过分析我们可以看出，票价低于50的乘客生存率最低，低于40%，其他票价区间的乘客的生存率都达到了60%以上，总体上票价越贵的乘客生存率更高。甚至，在唯有的三张500以上的奢侈舱位中，这三位乘客全部幸存了下来，生存率达到100%。

# 我猜测造成该生存率差异的原因是，更高票价的舱位可能拥有逃生的优先机会，同时高价舱位拥有完备的逃生设备和引导。低价舱位的救生条件不好，导致了生存率很低。

# 以上猜测也不能说明生存率与票价之间存在必然联系，低价舱位可能因为人数过于庞大，造成逃生时候拥堵，或导致踩踏事件，造成生存率很低。而高价舱位因为人数很少，不会因为拥堵而失去逃生的机会。当然，高价舱位本身数据量过小，也不具有代表性。

# # 五、结论阶段(Drawing Conclusions Phase)

# 总结以上各项分析，得出结论如下：

# 1、0至10岁年龄段的乘客有较高的生存率，60至70岁和20至30岁两个年龄段的生存率相对更低。

# 2、1级舱位、2级舱位、3级舱位的生存率逐个递减。

# 3、男性乘客的生存率仅有20%左右，女性乘客的生存率高达75%左右。男性乘客的生存率远远低于女性乘客，差距非常悬殊。

# 4、除了单独出行的乘客生存率较低，携带1位亲属的乘客生存率最高，携带1位亲属至5位亲属的乘客生存率逐个降低。

# 5、票价低于50的乘客生存率最低，低于40%，其他票价区间的乘客的生存率都达到了60%以上，总体上票价越贵的乘客生存率更高。

# 6、以上票价和各变量的联系并不是必然的，他们都有诸多干扰因素。因此，以上分析只是一种推测，并不具有绝对意义。

# # 六、沟通阶段(Communication Phase)

# 1、0至10岁年龄段的乘客有较高的生存率，60至70岁和20至30岁两个年龄段的生存率相对更低。原因分析：Titanic的船员在紧急情况下可能将救生艇更多地留给了儿童，乘客在面对生命的抉择时往往会舍弃自己而为孩子留下更多生存的机会；因为处于20至30岁和60和70岁年龄段的乘客大多是0至10岁那部分儿童乘客的父母或祖父母，他们可能在紧急时刻放弃了自己的生命，把机会留给了孩子。

# In[19]:


titanic_sur_rate_df.plot(kind = 'bar', title = 'survived rate at all ages')


# 2、1级舱位、2级舱位、3级舱位的生存率逐个递减。原因分析：更高级的舱位的乘客可能享有了优先逃生的机会，这可能是因为高级舱位提供了足够的逃生设备，或者高级舱位所处的位置更加安全，也有可能船员救生时为高级舱位的乘客提供了“绿色通道”。

# In[20]:


titanic_sur_pcalss_rate_df.plot(kind = 'bar', title = 'survived rate at every pclass')


# 3、男性乘客的生存率仅有20%左右，女性乘客的生存率高达75%左右。男性乘客的生存率远远低于女性乘客，差距非常悬殊。原因分析：可能是因为船员救生的时候更多地照顾了女性乘客，也有可能在灾难发生时， 大多男性乘客把生存的机会留给了自己的爱人(类似于《Titanic》的剧情，Jack把生存的机会留给了Rose)。

# In[21]:


titanic_sur_rate_sex_df.plot(kind = 'bar', title = 'survived rate at each gender')


# 4、除了单独出行的乘客生存率较低，携带1位亲属的乘客生存率最高，携带1位亲属至5位亲属的乘客生存率逐个降低。原因分析：可能单独出行的乘客需要独自完成自救，求生能力不高，而携带一位亲属的乘客在面对自己的一个家人时具有更强的求生欲望和求生能力，他们可以协作完成逃生；携带过多亲属的乘客会因为需要照顾太多的亲人而拖延求生时间，导致了他们生存率更低。

# In[22]:


titanic_sur_dead_SibSp_count_df.plot(kind = 'bar', title = 'survived rate at every SibSp')


# 5、票价低于50的乘客生存率最低，低于40%，其他票价区间的乘客的生存率都达到了60%以上，总体上票价越贵的乘客生存率更高。原因分析：更高票价的舱位可能拥有逃生的优先机会，同时高价舱位拥有完备的逃生设备和引导。低价舱位的救生条件不好，导致了生存率很低。

# In[23]:


titanic_sur_fare_rate_df.plot(kind = 'bar', title = 'survived rate at all fares')

