import matplotlib.pyplot as plt
import numpy as np
import pandas


fig=plt.figure()

fig.suptitle('Snickers vs Twix')

fig, ax_lst= plt.subplots(2,2)

for i in range(5)
    new=str(i+1),"st plot"
    ax_lst[0][0].set_title(new)



x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')

plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()

x=np.linspace(0,2*np.pi)
y1=np.sin(x)
y2=np.cos(x)

#1st plot

fig, ax_lst= plt.subplots(2,1)
fig.suptitle("Trig Funcs")

ax_lst[0].plot(x,y1)
ax_lst[0].set_title("Sine graph")

ax_lst[1].plot(x,y2)
ax_lst[1].set_title("Cosine graph")


#2nd plot
x=np.linspace(0,2*np.pi)
y1=np.sin(x)
y2=np.cos(x)

fig,ax=plt.subplots()
ax.plot(x,y1, label='Sine')
ax.plot(x,y2, label='Cosine')

ax.set_title('Trig Funcs')
ax.legend()

#histograms

x = np.random.normal(size = 1000)
plt.hist(x, density=True, bins=30)
plt.ylabel('Probability');


#bar charts

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()

#pie charts

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


