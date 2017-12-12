# ridge-regression

Пытаемся сделать линейную регрессию
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn.gif)

находим w через МНК ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(1).gif)

Может возникнуть проблема: обратная матрица ковариационной матрицы  ![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(2).gif) не берется (когда есть почти линейная комбинация признаков)

Решение: прибавим ко всем собственным значениям число С-гребневый коэф регрессии (C = 0.001)
![](https://raw.githubusercontent.com/okiochan/ridge-regression/master/CodeCogsEqn(3).gif)

Теперь обратная матрица сущ. Более того, С запрещает весам w иметь большие значения. 
