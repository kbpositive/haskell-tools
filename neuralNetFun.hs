matrix x y = [
    [
        j+i*x |
        j <- [1..x]] |
    i <- [0..(y-1)]]

vec_mul x y = [(x!!c)*(y!!c)| c <- [0..(length x)-1]]

vec_minus x y = [(x!!c)-(y!!c)| c <- [0..(length x)-1]]

mat_mul x y = [[sum (vec_mul a ([k!!b|k<-y])) | b <- [0..(length (y!!0))-1]] | a <- x]

input = [[1.0,2.0,3.0,1.0]]

weights = [
    [0.1,0.2,0.3],
    [0.4,0.5,0.2],
    [0.7,0.3,0.1],
    [0.5,0.2,0.4]
    ]

targets = [1.0,0.0,1.0]

output x y = [tanh i | i <- ((mat_mul x y) !! 0)]

output' x y = [tanh' i | i <- ((mat_mul x y) !! 0)]

output_s x y = [sigmoid i | i <- ((mat_mul x y) !! 0)]

output_s' x y = [sigmoid' i | i <- ((mat_mul x y) !! 0)]

e = 1 + sum [(1 / product [1..n]) | n <- [1..10]]

sigmoid x = 1 / (1 + e**(-x))

sigmoid' x = (sigmoid x)*(1 - (sigmoid x))

sigmoid_layer x = [sigmoid (x!!n) | n <- [0..(length x)-1]]

sigmoid'_layer x = [sigmoid' (x!!n) | n <- [0..(length x)-1]]

tanh' x = 1 - ((tanh x)**2)

tanh_layer x = [tanh (x!!n) | n <- [0..(length x)-1]]

tanh'_layer x = [tanh' (x!!n) | n <- [0..(length x)-1]]

err p t = ((p-t)**2)/2

err_layer p t = [err (p!!n) (t!!n) | n <- [0..(length p)-1]]

err' p t = p - t

err'_layer p t = [err' (p!!n) (t!!n) | n <- [0..(length p)-1]]

backprop inpts wgts tgts = [[i*j | j <- (vec_mul (err'_layer (output_s inpts wgts) tgts) (output_s' inpts wgts))] | i <- (inpts!!0)]

update inpts wgts tgts = [vec_minus (wgts!!n) ((backprop inpts wgts tgts)!!n) | n <- [0..(length wgts)-1]]

updateLoop :: Integer -> [[Double]]
updateLoop 0 = weights
updateLoop x = update input (updateLoop (x - 1)) targets