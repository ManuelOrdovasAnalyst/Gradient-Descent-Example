# Gradient-Descent-Example
Gradient descent example for multiple linear regression in Julia

## Introducción:

La técnica comúnmente llamada "Gradient descent" es un método moderno de estimación de parámetros empleado notablemente en el campo del Machine Learning. Aunque sea una técnica ampliamente utilizada, la documentación disponible para implementarla suele estar fragmentada o orientada a regresiones lineales simples. Este documento trata de exponer de forma clara el funcionamiento de esta técnica, enfocándose en la práctica y realizando un ejemplo práctico completo en el lenguaje Julia (Ya que creo que este lenguaje es el más comprensible, incluso para personas que no lo conozcan).

<hr>

### Conceptos básicos

La fórmula de una regresión lineal múltiple tiene la siguiente forma:

<p align=center>y = β0 + β1x1 + β2x2 + · · · + βmXm + ε</p>

</br></br>

¿Cómo se relaciona el contenido de una base de datos con esta formula?

* "**y**" es nuestra variable independiente

<p align=center><a  href="https://www.codecogs.com/eqnedit.php?latex=y=\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;...\\&space;y_{n}\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=\begin{bmatrix}&space;y_{1}\\&space;y_{2}\\&space;...\\&space;y_{n}\\&space;\end{bmatrix}" title="y=\begin{bmatrix} y_{1}\\ y_{2}\\ ...\\ y_{n}\\ \end{bmatrix}" /></a></p>

</br></br>

* "**β0 + β1x1 + β2x2 + · · · + βmXm + ε**" Es la suma de varias cosas:
   1. "**β0**": Hace referencia a una **constante**.
   2. "**βi**": Hace referencia al valor Beta que hace que la variable "**xi**" prediga "**yi**" de la mejor forma posible.
   3. "**xi**": Hace referencia al valor en una variable.
   4. "**ε**": Es el error, ya que todas las estimaciones no triviales tendrán un determinado error.

</br></br>

La constante a la que nos hemos referido anteriormente tiene como utilidad la de estimar cuál es el valor de **y** cuando el resto de variables es cero, y está compuesta por el valor desconocido de **β0**.

x1, x2, x3, ... hacen referencia a los valores que predicen **y** en una base de datos 

<hr>

### Una medida para el error

Usualmente la medida empleada es el Error Cuadrático Medio (**MSE**). Esta medida se obtiene al hacer la media de todos los errores al cuadrado (diferencia al cuadrado entre la **y** observada en la base y la **y** predicha). ¿Por qué al cuadrado? Porque como a veces el error será positivo y otras negativo, al elevarlo al cuadrado nos aseguramos que los valores negativos no anulan a los positivos.

La fórmula de este error es la siguiente:

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{n}\sum&space;(&space;y_{observada}&space;-&space;y_{predicha})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{n}\sum&space;(&space;y_{observada}&space;-&space;y_{predicha})^2" title="MSE = \frac{1}{n}\sum ( y_{observada} - y_{predicha})^2" /></a></p>

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=y_{predicha}=\beta0&space;&plus;&space;\beta1x1&space;&plus;&space;\beta2x2&space;&plus;&space;...&space;&plus;&space;\beta&space;mXm&space;&plus;&space;\varepsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{predicha}=\beta0&space;&plus;&space;\beta1x1&space;&plus;&space;\beta2x2&space;&plus;&space;...&space;&plus;&space;\beta&space;mXm&space;&plus;&space;\varepsilon" title="y_{predicha}=\beta0 + \beta1x1 + \beta2x2 + ... + \beta mXm + \varepsilon" /></a></p>

</br></br>

¿Cómo puedo saber los valores de **β**? Inicialmente son valores aleatorios que poco a poco se irán acercando a los finales. Es este proceso de irse aproximando a los valores finales lo que da a la técnica el nombre "**gradient descent**"

<hr>

### Aproximándonos a los valores reales de β

La derivada parcial del error nos proporciona en qué medida el error de nuestra predicción se aproxima al mínimo (0). Sin profundizar en la matemática, este cálculo es bastante sencillo:

<p align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dm}=\frac{2}{n}\sum&space;-x_{i}(y_{observada}-y_{predicha})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dm}=\frac{2}{n}\sum&space;-x_{i}(y_{observada}-y_{predicha})" title="\frac{d}{dm}=\frac{2}{n}\sum -x_{i}(y_{observada}-y_{predicha})" /></a></p>

</br></br>

Trataremos que este valor sea lo más prósimo a 0 que sea posible. (Observamos que al hacer la derivada parcial, el elevado al cuadrado desaparece, esto permite que este valor sea negativo o positivo)

</br></br>

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

</br></br>

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

## Continuará mañana, si quieres el código completo está aquí (funciona, aunque aún no lo he revisado con calma):

```Julia

using DataFrames, Distributions, StatsBase, Statistics

function sumario(resultado)
    println(" ")
    println(" ")
    println(" ")
    for i in resultado
        println("$(i[1]) : $(i[2])")
    end
end




function linearRegression(Ind, Deps, datos, tol = 1e-9, maxiter=100000, lr = 0.0001)

    converge = false
    contador = 0

    # y: variable independiente; x dependientes, m = n observaciones
    y     = reshape(datos[!,Symbol(Ind)],size(datos,1),1)
    x     = convert(Matrix, datos[!,[Symbol(i) for i in Deps]])
    theta = reshape(randn(size(x,2)),(size(x,2),1))
    m     = length(y)

    # Función con mi gradiente a minimizar
    gradient(x,y,theta) = (2/m) * -1* (transpose(x) * (y - x*theta))

    while !converge
        contador += 1
        theta2 = theta - lr * gradient(x,y,theta) # Ajuste de mis predictores
        if all((theta2 - theta).< tol)
            converge = true
        else
            if contador > maxiter
                error("No hay convergencia con $contador iteraciones")
            end
            theta = theta2
        end
    end

    # Cálculo de los valores T para cada predictor
    preds = x-y/theta # Diferencia entre los valores en la base y mis predictores
    err   = [std(preds[:,i]) for i in 1:size(preds,2)]
    gl    = m - (length(Deps)-1) - 1 # m - k (sin constante) -1
    Tvalores = theta./err
    Pvalores = [2*ccdf(TDist(gl),i) for i in Tvalores]
    res = DataFrame(Variables =Deps ,Beta=vcat(theta...),StdErr=err,Tvalue=vcat(Tvalores...),Pvalue=vcat(Pvalores...))

    # Cálculo de la significacion global del modelo (http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm)
    yPred = x*theta

    SSM(yPred) = sum((yPred.-mean(y)).^2)  # Suma de cuadrados corregidas para el modelo
    SSE(y,yPred) = sum((y-yPred).^2)       # Suma de cuadrados para el error
    SST(y) = sum((y.-mean(y)).^2)          # Suma de cuadrados corregidas para el total
    DFM = length(Deps)-1                   # Grados de libertad corregidos para el modelo
    DFE = m-length(Deps)                   # Grados de libertad del error
    MSM = SSM(yPred)/DFM                   # Media de los cuadrados para el modelo
    MSE = SSE(y,yPred)/(DFE)               # Media de cuadrados del error
    MST = SST(y)/(m-1)                     # Media de cuadrados del total

    F = (SSM(yPred)/(DFE))/(SSE(y,yPred)/(DFE))  # Valor del estadistico global F
    P = ccdf(FDist(DFM,DFE),F)                   # P valor

    Rsquared = cor(y,yPred)[1]
    RsquaredAdj = 1 - (1-Rsquared)*(m-1)/DFE
    resultado = Dict("1. Coefficients"=> res,"2. Ajuste del modelo" => "F = $F ; P valor = $P","3. Rsquared"=> Rsquared,"4. RsquaredAdj"=>RsquaredAdj)

    return resultado
end


n   = 1000
cte = [1 for i in 1:n]
X   = randn(n)
Z   = randn(n)
R   = randn(n)
Q   = randn(n)
Y   = (2) .+ (2 .*X) + (0 .*Q) + (3 .*Z) + (5 .*R) + (2*randn(n))

datos = DataFrame(Y=Y, cte=cte, X=X, Z=Z, R=R, Q=Q)
print(head(datos))


Ind = "Y"
Deps = ["cte","X","Q","Z","R"]
res = linearRegression(Ind, Deps, datos)
sumario(res)

```
