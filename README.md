# Gradient-Descent-Example
Gradient descent example for multiple linear regression in Julia

## Introducción:

La técnica comúnmente llamada "Gradient descent" es un método moderno de estimación de parámetros empleado notablemente en el campo del [**Machine Learning**](https://www.freecodecamp.org/news/understanding-gradient-descent-the-most-popular-ml-algorithm-a66c0d97307f/). Aunque sea una técnica ampliamente utilizada, la documentación disponible para implementarla suele estar fragmentada o orientada a regresiones lineales simples. Este documento trata de exponer de forma clara el funcionamiento de esta técnica, enfocándose en la práctica y realizando un ejemplo práctico completo en el lenguaje [Julia](https://julialang.org/) (Ya que creo que este lenguaje es el más comprensible, incluso para personas que no lo conozcan).

<hr>

### Conceptos básicos

La fórmula de una regresión lineal múltiple tiene la siguiente forma:

<p align=center>y = β0 + β1x1 + β2x2 + · · · + βmXm + ε</p>

</br>

¿Cómo se relaciona el contenido de una base de datos con esta formula?

* "**y**" es nuestra variable independiente

<p align=center><a  href="https://www.codecogs.com/eqnedit.php?latex=y=\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;...\\&space;y_{n}\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;...\\&space;y_{n}\\&space;\end{bmatrix}" title="y=\begin{bmatrix} y_{1}\\ y_{2}\\ ...\\ y_{n}\\ \end{bmatrix}" /></a></p>

</br>

* "**β0 + β1x1 + β2x2 + · · · + βmXm + ε**" Es la suma de varias cosas:
   1. "**β0**": Hace referencia a una **constante**.
   2. "**βi**": Hace referencia al valor Beta que hace que la variable "**xi**" prediga "**yi**" de la mejor forma posible.
   3. "**xi**": Hace referencia al valor en una variable.
   4. "**ε**": Es el error, ya que todas las estimaciones no triviales tendrán un determinado error.

</br>

La constante a la que nos hemos referido anteriormente tiene como utilidad la de estimar cuál es el valor de **y** cuando el resto de variables es cero, y está compuesta por el valor desconocido de **β0**.

x1, x2, x3, ... hacen referencia a los valores que predicen **y** en una base de datos 

<hr>

### Una medida para el error

Usualmente la medida empleada es el [Error Cuadrático Medio](https://es.wikipedia.org/wiki/Error_cuadr%C3%A1tico_medio) (**MSE**). Esta medida se obtiene al hacer la media de todos los errores al cuadrado (diferencia al cuadrado entre la **y** observada en la base y la **y** predicha). ¿Por qué al cuadrado? Porque como a veces el error será positivo y otras negativo, al elevarlo al cuadrado nos aseguramos que los valores negativos no anulan a los positivos.

La fórmula de este error es la siguiente:

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{n}\sum&space;(&space;y_{observada}&space;-&space;y_{predicha})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{n}\sum&space;(&space;y_{observada}&space;-&space;y_{predicha})^2" title="MSE = \frac{1}{n}\sum ( y_{observada} - y_{predicha})^2" /></a></p>

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=y_{predicha}=\beta0&space;&plus;&space;\beta1x1&space;&plus;&space;\beta2x2&space;&plus;&space;...&space;&plus;&space;\beta&space;mXm&space;&plus;&space;\varepsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{predicha}=\beta0&space;&plus;&space;\beta1x1&space;&plus;&space;\beta2x2&space;&plus;&space;...&space;&plus;&space;\beta&space;mXm&space;&plus;&space;\varepsilon" title="y_{predicha}=\beta0 + \beta1x1 + \beta2x2 + ... + \beta mXm + \varepsilon" /></a></p>

</br>

¿Cómo puedo saber los valores de **β**? Inicialmente son valores aleatorios que poco a poco se irán acercando a los finales. Es este proceso de irse aproximando a los valores finales lo que da a la técnica el nombre "**gradient descent**"

<hr>

### Aproximándonos a los valores reales de β

La [derivada parcial](https://es.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/introduction-to-partial-derivatives) del error nos proporciona en qué medida el error de nuestra predicción se aproxima al mínimo (0). Sin profundizar en la matemática, este cálculo para el **MSE** es bastante sencillo:

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dm}=\frac{2}{n}\sum&space;-x_{i}(y_{observada}-y_{predicha})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dm}=\frac{2}{n}\sum&space;-x_{i}(y_{observada}-y_{predicha})" title="\frac{d}{dm}=\frac{2}{n}\sum -x_{i}(y_{observada}-y_{predicha})" /></a></p>

</br>

Trataremos que este valor sea lo más prósimo a 0 que sea posible. (Observamos que al hacer la derivada parcial, el elevado al cuadrado desaparece, esto permite que este valor sea negativo o positivo)

</br>

Finalmente, idearemos la lógica para aproximar el valor de nuestras **β** inventadas a las reales. Este será un proceso iterativo que irá restando o sumando a nuestras **β** determinados valores (la derivada parcial previamente indicada). La cantidad de cambio que querremos que se aplique a cada paso se llama **tasa de aprendizaje** o **learning rate**, y es proporcionado a priori por nosotros. Establezco de forma arbitraria un **learning rate** (lr) de 0.0001.

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;repetir.hasta.convergencia&space;\begin{Bmatrix}&space;\beta_{1:4}:=\beta_{1:4}-lr\frac{d}{dm}\\&space;\end{Bmatrix}&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;repetir.hasta.convergencia&space;\begin{Bmatrix}&space;\beta_{1:4}:=\beta_{1:4}-lr\frac{d}{dm}\\&space;\end{Bmatrix}&space;\end{matrix}" title="\begin{matrix} repetir.hasta.convergencia \begin{Bmatrix} \beta_{1:4}:=\beta_{1:4}-lr\frac{d}{dm}\\ \end{Bmatrix} \end{matrix}" /></a></p>

<hr>

### Éxito en nuestro algoritmo

Que hallamos tenido éxito a la hora de predecir puede ser medido mediante distintos acercamientos. El aquí empleado es que la diferencia entre el valor de las **β** de una iteración con la siguiente sea muy pequeño (p.ej 1e-9).

También es conveniente fijar un número máximo de iteraciones que si el algoritmo supera indique que hemos fracasado (p.ej maxiter = 100000).

<hr>

### Nuestra base de datos

Tendrá forma rectangular, y dispondremos de una variable "**Y**" que querremos predecir a través de las variables "X, Z, R y Q":

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;Y&&space;X&space;&&space;Z&space;&&space;R&space;&&space;Q\\&space;2&&space;3&space;&&space;5&space;&&space;6&space;&&space;2\\&space;4&&space;4&space;&&space;2&space;&&space;1&space;&&space;6\\&space;...&...&space;&&space;...&space;&&space;...&space;&...&space;\\&space;9&&space;3&space;&&space;2&space;&&space;1&space;&&space;5&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;Y&&space;X&space;&&space;Z&space;&&space;R&space;&&space;Q\\&space;2&&space;3&space;&&space;5&space;&&space;6&space;&&space;2\\&space;4&&space;4&space;&&space;2&space;&&space;1&space;&&space;6\\&space;...&...&space;&&space;...&space;&&space;...&space;&...&space;\\&space;9&&space;3&space;&&space;2&space;&&space;1&space;&&space;5&space;\end{bmatrix}" title="\begin{bmatrix} y& x1 & x2 & x3 & x4\\ 2& 3 & 5 & 6 & 2\\ 4& 4 & 2 & 1 & 6\\ ...&... & ... & ... &... \\ 9& 3 & 2 & 1 & 5 \end{bmatrix}" /></a></p>

</br>

En Julia, utilizando el paquete DataFrames crearemos esta base de datos de la siguiente manera:

```Julia
using DataFrames

n   = 1000

cte = [1 for i in 1:n]
X   = randn(n)
Z   = randn(n)
R   = randn(n)
Q   = randn(n)
Y   = (2) .+ (2 .*X) + (0 .*Q) + (3 .*Z) + (5 .*R) + (2*randn(n))

datos = DataFrame(Y = Y, cte=cte, X = X, Q = Q, Z = Z, R = R)
```
Con esto, tendremos una base de datos con una variable "**Y**" que es igual a (2) + (2 veces X) + (0 veces Q) + (3 veces Z) + (5 veces R) + error.

La variable cte.. ¿Recuerdas que la constante es un valor determinado cuando el resto de variables es 0?, como la base de datos carece de ningún indicador de esto, lo añadimos nosotros. Es simplemente una columna con el valor constante 1.

Obtendremos una base de datos similar a la siguiente:

```
6×6 DataFrame
│ Row │ Y        │ cte   │ X         │ Z         │ R         │ Q         │
│     │ Float64  │ Int64 │ Float64   │ Float64   │ Float64   │ Float64   │
├─────┼──────────┼───────┼───────────┼───────────┼───────────┼───────────┤
│ 1   │ 13.3001  │ 1     │ 1.1557    │ 1.37189   │ 0.467395  │ 0.453063  │
│ 2   │ 7.21399  │ 1     │ 0.742289  │ 0.403971  │ 0.156844  │ -0.353715 │
│ 3   │ 0.784931 │ 1     │ 1.15784   │ 0.225592  │ -0.566034 │ 1.59162   │
│ 4   │ 11.1596  │ 1     │ -0.783623 │ 0.691487  │ 2.06183   │ -0.339737 │
│ 5   │ 1.80606  │ 1     │ -0.243656 │ -1.31121  │ 0.528422  │ 0.957272  │
│ 6   │ -1.1709  │ 1     │ -1.10038  │ -0.825377 │ -0.188085 │ 0.270251  │
```

<hr>

## Código para el análisis

Usualmente la mejor forma de aprender cómo funciona una técnica es verla en acción, los únicos requisitos relevantes necesarios para comprender el código es la [multiplicación de matrices](http://oceanologia.ens.uabc.mx/~matematicas/algebralineal/II%20Matrices/promat.htm) por vectores, esta operación funciona de la siguiente forma:

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=A=\begin{bmatrix}&space;a_{1,1}&a_{1,2}&a_{1,3}\\&space;a_{2,1}&a_{2,2}&a_{2,3}\\&space;a_{3,1}&a_{3,2}&a_{3,3}&space;\end{bmatrix},B=\begin{bmatrix}&space;b_{1}\\&space;b_{2}\\&space;b_{3}&space;\end{bmatrix};A\times&space;B=&space;\begin{bmatrix}&space;a_{1,1}*b_{1}&plus;a_{1,2}*b_{2}&plus;a_{1,3}*b_{3}\\&space;a_{2,1}*b_{1}&plus;a_{2,2}*b_{2}&plus;a_{2,3}*b_{3}\\&space;a_{3,1}*b_{1}&plus;a_{3,2}*b_{1}&plus;a_{3,3}*b_{3}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A=\begin{bmatrix}&space;a_{1,1}&a_{1,2}&a_{1,3}\\&space;a_{2,1}&a_{2,2}&a_{2,3}\\&space;a_{3,1}&a_{3,2}&a_{3,3}&space;\end{bmatrix},B=\begin{bmatrix}&space;b_{1}\\&space;b_{2}\\&space;b_{3}&space;\end{bmatrix};A\times&space;B=&space;\begin{bmatrix}&space;a_{1,1}*b_{1}&plus;a_{1,2}*b_{2}&plus;a_{1,3}*b_{3}\\&space;a_{2,1}*b_{1}&plus;a_{2,2}*b_{2}&plus;a_{2,3}*b_{3}\\&space;a_{3,1}*b_{1}&plus;a_{3,2}*b_{1}&plus;a_{3,3}*b_{3}&space;\end{bmatrix}" title="A=\begin{bmatrix} a_{1,1}&a_{1,2}&a_{1,3}\\ a_{2,1}&a_{2,2}&a_{2,3}\\ a_{3,1}&a_{3,2}&a_{3,3} \end{bmatrix},B=\begin{bmatrix} b_{1}\\ b_{2}\\ b_{3} \end{bmatrix};A\times B= \begin{bmatrix} a_{1,1}*b_{1}+a_{1,2}*b_{2}+a_{1,3}*b_{3}\\ a_{2,1}*b_{1}+a_{2,2}*b_{2}+a_{2,3}*b_{3}\\ a_{3,1}*b_{1}+a_{3,2}*b_{1}+a_{3,3}*b_{3} \end{bmatrix}" /></a></p>

</br>

Podemos observar que el resultado es un vector que contiene la suma de los valores de las columnas multiplicados por los valores del vector. Si considerásemos que nuestro vector es un conjunto de valores theta y nuestra matriz es nuestra base de datos, al realizar este cálculo obtendríamos los valores de **y** estimados para cada sujeto.

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=X=\begin{bmatrix}&space;x_{1,1}&x_{1,2}&x_{1,3}\\&space;x_{2,1}&x_{2,2}&x_{2,3}\\&space;x_{3,1}&x_{3,2}&x_{3,3}&space;\end{bmatrix},Thetas=\begin{bmatrix}&space;\Theta&space;_{1}\\&space;\Theta_{2}\\&space;\Theta_{3}&space;\end{bmatrix};X\times&space;Thetas=&space;\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;y_{3}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X=\begin{bmatrix}&space;x_{1,1}&x_{1,2}&x_{1,3}\\&space;x_{2,1}&x_{2,2}&x_{2,3}\\&space;x_{3,1}&x_{3,2}&x_{3,3}&space;\end{bmatrix},Thetas=\begin{bmatrix}&space;\Theta&space;_{1}\\&space;\Theta_{2}\\&space;\Theta_{3}&space;\end{bmatrix};X\times&space;Thetas=&space;\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;y_{3}&space;\end{bmatrix}" title="X=\begin{bmatrix} x_{1,1}&x_{1,2}&x_{1,3}\\ x_{2,1}&x_{2,2}&x_{2,3}\\ x_{3,1}&x_{3,2}&x_{3,3} \end{bmatrix},Thetas=\begin{bmatrix} \Theta _{1}\\ \Theta_{2}\\ \Theta_{3} \end{bmatrix};X\times Thetas= \begin{bmatrix} y_{1}\\ y_{2}\\ y_{3} \end{bmatrix}" /></a></p>

</br>

Tras esta breve introducción, comenzaremos con el código para la realización de este procedimiento:

Nuestra base de datos:
```Julia
cte = [1 for i in 1:n]
X   = randn(n)
Z   = randn(n)
R   = randn(n)
Q   = randn(n)

Y   = (2) .+ # Theta de la constante = 2
   (2 .*X) + # Theta de X            = 2
   (0 .*Q) + # Theta de Q            = 0 (no ayudará a predecir Y)
   (3 .*Z) + # Theta de Z            = 3
   (5 .*R) + # Theta de R            = 5 (variable de mayor peso)
   (2*randn(n)) # error

datos = DataFrame(Y = Y, cte=cte, X = X, Q = Q, Z = Z, R = R)

│ Row │ Y        │ cte   │ X         │ Z         │ R         │ Q         │
│     │ Float64  │ Int64 │ Float64   │ Float64   │ Float64   │ Float64   │
├─────┼──────────┼───────┼───────────┼───────────┼───────────┼───────────┤
│ 1   │ 13.3001  │ 1     │ 1.1557    │ 1.37189   │ 0.467395  │ 0.453063  │
│ 2   │ 7.21399  │ 1     │ 0.742289  │ 0.403971  │ 0.156844  │ -0.353715 │
..........................................................................
```
</br>
Trataremos de recuperar los valores de **Theta** en el código anterior presentados a través de "**Gradient descent**"

```Julia
using DataFrames

Ind = "Y"                                   # Variable independiente (a predecir)
Deps = ["cte","X","Q","Z","R"]              # Variables dependientes (que predicen)

function gradientDescent(Ind, Deps, datos, tol = 1e-9, maxiter=100000, lr = 0.0001)
    converge = false  # Inicialmente, el algoritmo no ha convergido
    contador = 0      # Valor del contador de iteraciones inicial = 0
    
    # y: variable independiente; x dependientes, m = n observaciones
    y     = reshape(datos[!,Symbol(Ind)],size(datos,1),1)
    x     = convert(Matrix, datos[!,[Symbol(i) for i in Deps]])
    theta = reshape(randn(size(x,2)),(size(x,2),1)) # Establecemos thetas iniciales aleatorios (media = 0; std = 1)
    #n: numero de observaciones; m: numero de variables dependientes
    n     = length(y)
    m     = length(Deps)
    # Función con mi gradiente a minimizar (derivada parcial del error)
    gradient(x,y,theta) = (2/n) * -1* (transpose(x) * (y - x*theta))
    
    while !converge                               # Mientras no haya convergencia
    
        theta2 = theta - lr * gradient(x,y,theta) # Ajuste de mis predictores (acerca theta a los valores que menos error generan. es importante destacar que este paso debe realizarse para todos los thetas simultáneamente)
        
        if all((theta2 - theta).< tol) # compruebo si la diferencia de todos los thetas anteriores con los actuales sea menor con el valor que he tomado como referencia. Si lo es, el algoritmo ha convergido
            converge = true
        else                                     # Si no ha convergido
            if contador > maxiter                # Compruebo si ha realizado tantas iteraciones como el máximo propuesto (maxiter). De ser así, el algoritmo no ha convergido.
                error("No hay convergencia con $contador iteraciones")
            else # Si ni ha convergido ni ha alcanzado el número máximo de iteraciones, realiza una iteración más (representado en el contador), y el valor theta se actualiza.
                contador += 1
                theta = theta2
            end
        end
    end
    # Una vez el algoritmo ha convergido, genera un DataFrame con los valores de las variables dependientes y de los thetas
    return DataFrame(Dependientes = Deps,thetas = vcat(theta...)) #DataFrame precisa de vectores para generar columnas (vcat nos sirve con este propósito)
end

```
Y ya está!

<hr>

#### Resultado:
```
thetas = gradientDescent(Ind, Deps, datos)
thetas.thetas = [round(i,digits=1) for i in thetas.thetas]
print(thetas)

5×2 DataFrame
│ Row │ Dependientes │ thetas  │
│     │ String       │ Float64 │
├─────┼──────────────┼─────────┤
│ 1   │ cte          │ 2.0     │
│ 2   │ X            │ 2.0     │
│ 3   │ Q            │ 0.1     │
│ 4   │ Z            │ 3.1     │
│ 5   │ R            │ 5.0     │
```

Como podemos observar, el valor de las thetas estimadas es MUY cercano al valor de las thetas reales. EXITO!
