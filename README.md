# ridge-regression

Пытаемся сделать линейную регрессию
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn.gif)

находим w через МНК ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(1).gif)

Может возникнуть проблема: обратная матрица ковариационной матрицы  ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(2).gif) не берется (когда есть почти линейная комбинация признаков)

Решение: прибавим ко всем собственным значениям ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(2).gif) число С-гребневый коэф регрессии (C = 0.001)
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(3).gif)

Теперь обратная матрица сущ. Более того, С запрещает весам w иметь большие по модулю значения. 

# Example

сгенирируем рандомную выборку, "продублируем ее справа" - получим выборку с линейной зависимостью 

вычислим ее ков. матрицу и ее собственные значения. видим, что есть нулевые.
... 1

Разберемся с проблемой: зададим С, вычислим новую ков. мартицу
```
c = 0.001
Cov2 = Cov + c * np.eye(n)
```
Теперь можем получить собств. знач и искомое решение W
