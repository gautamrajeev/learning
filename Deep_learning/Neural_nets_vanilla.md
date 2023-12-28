# Vanilla Neural Net (multi layer perceptron) : 

Learnt from this [link](https://www.3blue1brown.com/lessons/neural-networks) , this [book](http://neuralnetworksanddeeplearning.com/chap1.html)




## Simple structure 
Imagine that the number is represented by a pixel of 784 dots ( 28 *28 square). That details of how much each pixel is lit (ranging from 0-1 .. i.e 0.1,0.2, 0.9 etc) is fed to the first layer which has 784 nodes.
The output layer has 10 nodes (0-9). The hidden layers may contain any number of nodes. The final node whichever is most well lit, is the final prediction for which digit it is.  

Activation of the first layer leads to activation of the 2nd layer and so on until some combination of activation of the 3rd layer leads to the activation of the last layer.  

<img width="371" alt="image" src="https://github.com/gautamrajeev/learning/assets/86904775/f4e8f6b2-b5b4-4114-a03c-f8e74181d70e">

### Activation from first layer to seconds (weighting and biasing) 

#### Detecting an edge : 
Imagine now that the middle layers are for recognizing some part of the grid is lit or not :
For example : 

<img width="133" alt="image" src="https://github.com/gautamrajeev/learning/assets/86904775/b0f349c4-7c19-43dc-abb3-aa5c6c7be78e">

Now, to determine that that part is lit, it uses simple weights from a section of the previous layer. 
Intuitively thinking, it wants the pixels where the edge should be to be well lit and the surrounding pixels to be dark.  
So, the pixels where the edge should have +ve weights and the pixels where no edge should have -ve weights (because they are lit, then the image cannot be an edge) 

So that layer's node(neuron) gets lit depending on if the previous layers neuros activation weighted by a certain function. 
The next node may have a completely differnt set of functions guiding if they are getting lit or not.  Technically all nodes are lighting the next nodes in some combination of weights, some of these weigths may be 0 


**Sigmoid function** :  The weights can add up to a number not in between 0-1. So you use a Sigmoid function to squish them to  0-1. You also recently use ReLu { if x>50, then 1 else 0 - for binary activations } 

**Biasing** :  Sometimes, we don't want a neuron to light up easily. We only want the neuron to light up if the weighted sum is more than 50. So you add a bias to each neuron before sigmoidification 

So thats just for single neuron, for a 2nd layer with 16 neurons we have  that’s 784x16 weights and 16 biases.

Representing this as a matrix of weights, where $a^{(0)}$ is a matrix of the activations of the first layer and $a^{(1)}$ is what happens to the weights in the 2nd layer

$$ \begin{bmatrix}
    a_{0}^{(0)} &  a_{1}^{(0)} & \cdots &   a_{768}^{(0)}
\end{bmatrix}
\begin{bmatrix}
    w_{0,0} & w_{0,1} & \cdots & w_{0,16} \\
    w_{1,0} & w_{1,1} & \cdots & w_{1,16} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    w_{768,0} & w_{768,1} & \cdots & w_{768,16}
\end{bmatrix} 
+
\begin{bmatrix}
    b_{0} &     b_{1} &   \cdots &     b_{16}
\end{bmatrix} 
 =  \begin{bmatrix}
     a_{0}^{(1)} &  a_{1}^{(1)} & \cdots &   a_{16}^{(1)}
\end{bmatrix} 
$$


### Prediction: 

Showing the prediction method in terms of matrices: 

Imagine you have 200 images i.e. 200 data points and you have a single layer from 768 to 10 

Hence you have $A_ni$   as : 

$$ \begin{bmatrix}
    a_{0,0} & a_{0,1} & \cdots & a_{0,768} \\
    a_{1,0} & a_{1,1} & \cdots & a_{1,768} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    a_{200,0} & a_{768,1} & \cdots & a_{768,200}
\end{bmatrix} $$


where 
n :  number of data points 
i :  number of neurons in input layer :  number of pixels = 1-768

We have our weights from one layer to another $Wij$ 

$$ \begin{bmatrix}
    w_{0,0} & w_{0,1} & \cdots & w_{0,10} \\
    w_{1,0} & w_{1,1} & \cdots & w_{1,10} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    w_{768,0} & w_{768,1} & \cdots & w_{768,10}
\end{bmatrix} $$


where 
i :  number of neurons in input layer :  number of pixels = 1-768
j :  number of neurons in output layer :  1-10 

This will give us the final predicted output as 

$$ \begin{bmatrix}
    f(a_{1,i}*w_{i,1}) & f(a_{1,i}*w_{i,2}) & \cdots & f(a_{1,i}*w_{i,10})\\  
    f(a_{2,i}*w_{i,1}) & f(a_{2,i}*w_{i,2})  & \cdots & f(a_{2,i}*w_{i,10}) \\ 
     \vdots  & \vdots  & \ddots & \vdots  \\
      f(a_{200,i}*w_{i,1}) & f(a_{200,i}*w_{i,2})  & \cdots & f(a_{200,i}*w_{i,10}) 
\end{bmatrix} $$

where f(a*w) = Sigmoid(a * w + b)  


or
predicted output $P_{nj}$=   f($A_{ni}$ * $W_{ij}$) 


### Cost functions :  

At first, you intitialize all the weights randomly and then for all data points get outputs that are bad. You decide to have a cost function that tells you how bad is it and how you need to improve on it

Ideally cost function should be an absolute difference between expected output and actual output. However, we dont in reality do this because its not differentiable easliy.  We end up using 
(Expected_output - Predicted_output)^2 because this is differentiable.  

$C(P,E) = (P_{nj} - E_{nj})^2 $

The slope of this vector ($C_{nj}) is called the “gradient” symbolized by $, and it tells you which direction you should step to increase the function most quickly. Taking the negative of that vector gives you the direction to step which decreases the function most quickly. The length of that gradient vector is an indication for just how steep that steepest slope is.

