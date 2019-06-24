## Run
Firstly you have to install the dependencies.

    pip install Pillow
    
Then, run the following command to execute the algorithm.

    python main.py
    
## Background material in maths

We have the above **mixture distribution**:

![](http://latex.codecogs.com/gif.latex?%24%24%20P%28%5Cmathbf%7Bx%7D%29%20%3D%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20e%5E%7B-%5Cfrac%7B1%7D%7B2%20%5Csigma_k%5E2%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%7D%20%24%24)

<br>

We want to maximize the **log Likelihood**:

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%20%5Cmathcal%7BL%28M%2C%20%5Cpi%2C%20%5Csigma%29%7D%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20%5Clog%20p%28x_n%7CM%2C%20%5Cpi%2C%20%5Csigma%29%5C%3E%20%5Ctextbf%7Bfor%7D%20%5C%3E%20M%20%3D%20%5C%7B%5Cmu_%7Bk%2Cd%7D%5C%7D_%7Bk%3D1%2Cd%3D1%7D%5E%7BK%2CD%7D%20%5C%3E%20%5Ctextbf%7Band%7D%20%5C%3E%20%5Cpi%20%3D%20%5C%7B%5Cpi_k%5C%7D_%7Bk%3D1%7D%5E%7BK%7D%20%24)

<br>

We have to construct the **Complete likelihood**: 

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20P%28X%2C%20Z%7CM%2C%20%5Cpi%2C%20%5Csigma%29%20%3D%5Cprod_%7Bn%3D1%7D%5EN%20%5Cprod_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%5CBigg%5B%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20e%5E%7B-%5Cfrac%7B1%7D%7B2%20%5Csigma_k%5E2%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%7D%20%5CBigg%5D%5E%7Bz_%7Bn%2Ck%7D%7D%20%24%24)

<br>

And then the **Complete logLikelihood** is defined as:

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20L_c%28M%2C%20%5Cpi%2C%20%5Csigma%29%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20%5Csum_%7Bk%3D1%7D%5EK%20z_%7Bn%2Ck%7D%20%5Cbigg%5C%7B%20%5Clog%20%5Cpi_k%20&plus;%20%5Csum_%7Bd%3D1%7D%5ED%20%5Clog%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20-%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Csigma_k%5E2%7D%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%20%5Cbigg%5C%7D)

<br>

But we don't know the value of the hidden variable *z* each time, so we must compute the **Expected Complete likelihood**.
Since each ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%20z_%7Bn%2Ck%7D%20%24) appears linearly in ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%20L_%7Bc%7D%20%24) the mean value of ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%20L_%7Bc%7D%20%24) is obtained by substituting the ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%20z_%7Bn%2Ck%7D%20%24) with their mean value :

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20%5Cmathcal%7BQ%28M%2C%20%5Cpi%2C%20%5Csigma%29%7D%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%5Cbig%5C%7B%20%5Clog%20%5Cpi_k%20&plus;%20%5Csum_%7Bd%3D1%7D%5ED%20%5Clog%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20-%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Csigma_k%5E2%7D%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%20%5Cbig%5C%7D%20%24%24)

where 

<br>

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%3D%20%5Cfrac%7B%5Cpi_k%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20e%5E%7B-%5Cfrac%7B1%7D%7B2%20%5Csigma_k%5E2%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%7D%7D%7B%5Csum_%7Bj%3D1%7D%5EK%20%5Cpi_j%20%5Cprod_%7Bd%3D1%7D%5ED%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%20%5Cpi%20%5Csigma_k%5E2%7D%7D%20e%5E%7B-%5Cfrac%7B1%7D%7B2%20%5Csigma_k%5E2%7D%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%7D%7D%20%24%24)

In order to get the equations of the **Maximization Step**, we differentiate the above function with respect to ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%5Cmu_%7Bk%2Cd%7D%24) and ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%5Cpi_k%24) each time and assuming that ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%5Cgamma%28z_%7Bn%2Ck%7D%29%24) are constant (in terms of ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%5Cmu_%7Bk%2Cd%7D%24) and ![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%5Cpi_k%24)) since they have been calculated in the **Expectation Step** based on the previous parameter values. 
After that we set the differentiation equals to 0 :

<br>

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Cmu_%7Bk%2Cd%7D%7D%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%5CBigg%5B%5Cfrac%7B2%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%7D%7B%5Csqrt%7B2%20%5Csigma_k%5E2%7D%7D%20%5CBigg%5D%20%3D%20%5Cfrac%7B2%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%7B%5Csqrt%7B2%20%5Csigma_k%5E2%7D%7D%24%24)

<br>

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Cmu_%7Bk%2Cd%7D%7D%20%3D%200%20%5CRightarrow%20%28x_d%20-%20%5Cmu_%7Bk%2Cd%7D%29%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%3D%200%20%5CRightarrow%20%5Cmu_%7Bk%2Cd%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20x_d%7D%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%24%24)

<br>

On this step we need to add the LaGrange Multiplier in order to find the exact formula.

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Csmall%20%24%24%20%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%28Q%20&plus;%20%5Clambda%28%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_%7Bk%7D-1%29%29%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%7B%5Cpi_k%7D%20&plus;%20%5Clambda%24%24)

![](http://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Ctiny%20%24%24%20%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%20%3D%200%20%5CRightarrow%20%5Clambda%20%3D%20-%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%7B%5Cpi_%7Bk%7D%7D%20%5CLeftrightarrow%20%5Cpi_%7Bk%7D%5Clambda%20%3D%20%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%5CRightarrow%20%5Csum_%7Bj%3D1%7D%5EK%20%5Cpi_%7Bj%7D%5Clambda%20%3D%20%5Csum_%7Bj%3D1%7D%5EK%20%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%5CRightarrow%20%5Clambda%20%3D%20-N%24%24)

![](http://latex.codecogs.com/gif.latex?%5Csmall%20%24%24%20%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%20%3D%200%20%5CRightarrow%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%7B%5Cpi_%7Bk%7D%7D%20-%20N%20%3D%200%20%5CRightarrow%20%5Cpi_%7Bk%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%7BN%7D%24%24)

<br>

And last one:

![](http://latex.codecogs.com/gif.latex?%5Csmall%20%24%24%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20%5Csigma_%7Bk%7D%5E2%7D%20%3D%200%20%5CRightarrow%20%5Ccdots%20%5CRightarrow%20%5Csigma_%7Bk%7D%5E2%20%3D%20%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5EN%20%5Csum_%7Bd%3D1%7D%5ED%20%5Cgamma%28z_%7Bn%2Ck%7D%29%20%28x_%7Bn%2Cd%7D%20-%20%5Cmu_%7Bk%2Cd%7D%29%5E2%7D%7BD%20%5Csum_%7Bd%3D1%7D%5ED%20%5Cgamma%28z_%7Bn%2Ck%7D%29%7D%2C%20k%3D1%2C...K%20%24%24)

<br>

### Results of Experiments - Comparisons with the original Image (Max_steps = 100)
---

<h3><center>segments=1, Total error: 0.1746</center></h3>
<img src=images/1.png> 

<br>

<h3><center>segments=2, Total error: 0.0486</center></h3>
<img src=images/2.png>

<br>

<h3><center>segments=4, Total error: 0.0167</center></h3>
<img src=images/4.png> 

<br>

<h3><center>segments=8, Total error: 0.0078</center></h3>
<img src=images/8.png>

<br>

<h3><center>segments=16, Total error: 0.0044</center></h3>
<img src=images/16.png> 

<br>

<h3><center>segments=32, Total error: 0.0016</center></h3>
<img src=images/32.png>

<br>

<h3><center>segments=64, Total error: 0.0009</center></h3>
<img src=images/64.png> 





