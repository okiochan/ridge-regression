# ridge-regression

Пытаемся сделать линейную регрессию
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/formula/CodeCogsEqn.gif)

находим w через МНК ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/formula/CodeCogsEqn(1).gif)

Может возникнуть проблема: обратная матрица ковариационной матрицы  ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/formula/CodeCogsEqn(2).gif) не берется (когда есть почти линейная комбинация признаков)

Решение: прибавим ко всем собственным значениям ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/formula/CodeCogsEqn(2).gif) число С-гребневый коэф регрессии (C = 0.001)
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/formula/CodeCogsEqn(3).gif)

Теперь обратная матрица сущ. Более того, С запрещает весам w иметь большие по модулю значения. 

# example

сгенирируем рандомную выборку, "продублируем ее справа" - получим выборку с линейной зависимостью 

вычислим ее ков. матрицу и ее собственные значения. видим, что есть нулевые.

![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/img/1.png)

Разберемся с проблемой: зададим С, вычислим новую ков. мартицу
```
С = 0.001
Cov2 = Cov + С * np.eye(n)
```
Теперь можем получить собств. знач и искомое решение W

![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/img/2.png)

код программы в **rigde.py**

Построим	график	зависимости	функционала	качества	от	параметра	регуляризации (линии - это полученные веса, взятые по модулю).
Этот пример также показывает полезность применения регрессии Риджа к плохо обусловленным матрицам.

выборки в файле **dataRidge.py**, код в **ridge_graphic.py**
```
X, Y = dataRidge.DataBuilder().Build("ski")
```

![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/img/r1.png)