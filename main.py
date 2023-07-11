from functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from openpyxl import Workbook
import numpy as np
from scipy.stats import norm


def laplace(x):
    return round(norm.cdf(x), 2)

wb = Workbook()
ws = wb.active




while True:
    variant = int(input('Введите способ ввода данных: \n 1. Через консоль\n 2. Через файл\n'))
    if variant == 1:
        n = int(input('Введите количество точек:'))
        x = list(map(float, input('Введите через пробел значения X: ').split()))
        y = list(map(float, input('Введите через пробел значения Y: ').split()))
        break
    elif variant == 2:
        filename = 'temp.txt'  # имя файла, содержащего данные

        # Открываем файл на чтение
        with open(filename, 'r') as file:
            lines = file.readlines()  # считываем все строки из файла
            n = int(lines[0])  # первое число в файле - это количество чисел n
            x = list(map(float, lines[1].split()))  # разделяем вторую строку на числа
            y = list(map(float, lines[2].split()))  # разделяем третью строку на числа
        break
    else:
        print("Выбранный вариант отсутвует. Попробуйте ещё раз.")

c = stable_data(x, y, n)
x_min, y_min = c.Minn()
x_max, y_max = c.Maxx()
Rx, Ry = c.razmah()
r = round(c.col_integ())
hx, hy = c.lens_integ()
hx, hy = round(hx, 2), round(hy, 2)
a, b = c.gran_integ()
print('Минимальное значение X: ', x_min)
print('Минимальное значение Y: ', y_min)
print('Максимальное значение X: ', x_max)
print('Максимальное значение Y: ', y_max)
print('Размах значений X: ', Rx)
print('Размах значений Y: ', Ry)
print('Количество интервалов: ', r)
print('Длина интервала X: ', hx)
print('Длина интервала Y: ', hy)
print()
print('Границы интервалов:')
for i, j in enumerate(a):
    print(f'a{i} = ', j)
for i, j in enumerate(b):
    print(f'b{i} = ', j)
y1 = pd.Series(x)
x1 = pd.Series(y)
correlation = y1.corr(x1)
plt.scatter(x1, y1)
plt.plot(a[1], b[1], linestyle='dashed')
#plt.show()
plt.savefig('Кореляционное поле.png')
Ox = []
Oy = []

for i in range(1, 9):
    Ox.append(round(((a[i-1]+a[i])/2), 2))
    Oy.append(round(((b[i-1]+b[i])/2), 2))
#print(Ox, Oy)

sp_table = [['Yi / Xi'] + Ox + ['сумма построчная ( Ni для Y )']]
for i in range(0, 8):
    sp_table.append([Oy[i], 0, 0, 0, 0, 0, 0, 0, 0, 0])
count = 0
count1 = 0

for j in range(0, len(Ox)):
    for z in range(0, len(Oy)):
        temp = False
        for i in range(n):
            x10, y10 = x[i], y[i]
            if (x10-Ox[j-1])*(x10-Ox[j]) <= 0 and (y10-Oy[z-1])*(y10-Oy[z]) <= 0:
                sp_table[z+1][j+1] += 1

sp_tabletemp= []
sp_tableend= [['Yi / Xi'] + Ox + ['сумма построчная ( Ni для Y )']]
for i in sp_table[1:]:
    sp_tabletemp.append(i[::-1][:-1])
count = 0
for i in sp_tabletemp[::-1]:
    sp_tableend.append([Oy[count]] + i)
    count += 1
sp_tableend[-1][0] = 'сумма постолбцо-вая ( Ni для X )'
for subarray in sp_tableend:
    ws.append(subarray)
ws.append([])
sp_tableend1 = [['Xi'] + Ox[:-1], ['Ni'] + sp_tableend[-1][2:-1]]
sp_tableend2 = [['Yi'] + Oy[:-1]]
sp11 = ['Ni']
for i in range(1, len(sp_tableend)-1):
    sp11.append(sp_tableend[i][-1])
sp_tableend2.append(sp11)
   
for subarray in sp_tableend1:
    ws.append(subarray)

ws.append([])  
for subarray in sp_tableend2:
    ws.append(subarray)





# 2 лаба

sp_tableend3 = [['Интервалы', 'Середины интервалов', 'Абсолютная', 'Относительная', 'Накопленная',	'Эмпирическая функция распределения', 'Эмпирическая плотность распределения']]
counter = 0
for i in range(1, len(b)):
    try:
        counter += sp_tableend2[1][i]
    except:
        break
    strr = [f'({round(b[i-1], 2)}-{round(b[i], 2)})', round((b[i]+b[i-1])/2,2), sp_tableend2[1][i], sp_tableend2[1][i]/50, counter, counter/50, round(sp_tableend2[1][i]/50/hy, 2)]
    sp_tableend3.append(strr)
ws.append([])  
for subarray in sp_tableend3:
    ws.append(subarray)


sp1 = []
sp2 = []
for i in sp_tableend3[1:]:
    sp1.append(i[1])
    sp2.append(i[-2])
plt.clf()
plt.hist(sp1, histtype='step', cumulative=True, bins=10)
plt.savefig('foo1.png')


plt.clf()

sp111 = []
sp222 = []
for i in b:
    sp111.append(i)

for i in range(1, len(sp_tableend3)):
    sp222.append(sp_tableend3[i][-1])

# seaborn histogram
plt.clf()
l = len(sp222)
plt.bar(sp111[:l], sp222)

# Построение графика плотности распределения
#sns.kdeplot(sp222, color='red')
plt.plot(sp111[:l], sp222)
plt.savefig('foo2.png')

# 3 лаба
# , , '
sp_tableend4 = [['(b(i-1);bi)', 'yi', 'ni', 'Ni', 'pi*=ni/n', 'yi*pi*']]
yb = 0
pis = 0
ni = 0


for i in sp_tableend3[1:]:
    sp_tableend4.append([i[0], i[1], i[2], i[4], round(i[3], 2), round(i[1]*i[3], 2)])
    yb += i[1]*i[3]
    pis += i[3]
    ni += i[2]
sp_tableend4[0].append('yi-yв')
sp_tableend4[0].append('(yi-yв)^2*pi*')
sp_tableend4[0].append('(yi-yв)^3*pi*')
sp_tableend4[0].append('(yi-yв)^4*pi*')
m2, m3, m4 = 0, 0, 0
for i, j in enumerate(sp_tableend4[1:]):
    sp_tableend4[i+1].append(round(sp_tableend4[i+1][1]-yb,2))
    sp_tableend4[i+1].append(round((sp_tableend4[i+1][1]-yb)**2*sp_tableend4[i+1][4],2))
    sp_tableend4[i+1].append(round((sp_tableend4[i+1][1]-yb)**3*sp_tableend4[i+1][4],2))
    sp_tableend4[i+1].append(round((sp_tableend4[i+1][1]-yb)**4*sp_tableend4[i+1][4],2))
    m2 += round((sp_tableend4[i+1][1]-yb)**2*sp_tableend4[i+1][4],2)
    m3 += round((sp_tableend4[i+1][1]-yb)**3*sp_tableend4[i+1][4],2)
    m4 += round((sp_tableend4[i+1][1]-yb)**4*sp_tableend4[i+1][4],2)
sp_tableend4.append(['Сумма:', ' ', ni, ' ', pis, yb, ' ', m2, m3, m4])
ws.append([])  
for subarray in sp_tableend4:
    ws.append(subarray)

Db = m2
sugb = sqrt(Db)
Ab = m3/(sugb**3)
S2 = (n / (n-1))*Db
S = sqrt(S2)
Eb = m4/(sugb**4) - 3
Astar = sqrt(n*(n-1))/(n-2) * Ab
#Estar = (n-1)/((n-2)(n-3))*((n+1)*Eb + 6)
V = (S/yb)*100

ltemp = len(b) // 2 + 1
intermodmax = b[ltemp]
intermodmin = b[ltemp - 1]
M0 = intermodmin + (intermodmax-intermodmin)*(sp_tableend4[ltemp][2] - sp_tableend4[ltemp-1][2])/((sp_tableend4[ltemp][2] + sp_tableend4[ltemp-1][2])+(sp_tableend4[ltemp][2] + sp_tableend4[ltemp+1][2]))
ME = intermodmin + (intermodmax-intermodmin)*((n/2-sp_tableend4[ltemp][3])/sp_tableend4[ltemp][2])


# 5 лаба


sp_tableend5 = [['(b(i-1);bi)', 'ni', 'Ui-1', 'Ui', 'Ф(Ui-1)', 'Ф(Ui)', "pi'", "ni'", "(ni-ni')^2/ni'"]]
x1000 = 0
x2000 = 0
xnabl = 0
for i, j in enumerate(sp_tableend4[1:-1]):
    x1000 += laplace(round((b[i]-yb)/S, 2))-laplace(round((b[i-1]-yb)/S, 2))
    x2000 += n*laplace(round((b[i]-yb)/S, 2))-laplace(round((b[i-1]-yb)/S, 2))
    xnabl += ((float(j[2])-n*laplace((b[i]-float(yb))/S)-laplace((b[i-1]-float(yb))/S))**2)/n*laplace((b[i]-float(yb))/S)-laplace((b[i-1]-float(yb))/S)
    sp_t = [j[0], j[2], round((b[i-1]-yb)/S, 2), round((b[i]-yb)/S, 2), laplace((b[i-1]-yb)/S), laplace((b[i]-yb)/S), laplace((b[i]-yb)/S)-laplace((b[i-1]-yb)/S), n*laplace((b[i]-yb)/S)-laplace((b[i-1]-yb)/S),  round(((j[2]-n*laplace((b[i]-yb)/S)-laplace((b[i-1]-yb)/S))**2)/n*laplace((b[i]-yb)/S)-laplace((b[i-1]-yb)/S), 2)]
    sp_tableend5.append(sp_t)
sp_tableend5.append(['Сумма', ' ', ' ', ' ', ' ', ' ', x1000, x2000, xnabl])

ws.append([])  
for subarray in sp_tableend5:
    ws.append(subarray)

wb.save('sample.xlsx')