module Classifier
  ( matrix,
    vec_mul,
    vec_minus,
    mat_mul,
    weights,
    targets,
    e,
    output,
    output',
    output_s,
    output_s',
    err,
    err',
    err_layer,
    err'_layer,
    sigmoid,
    sigmoid',
    sigmoid_layer,
    sigmoid'_layer,
    tanh',
    tanh_layer,
    tanh'_layer,
    backprop,
    update,
    updateLoop,
    fit,
  )
where

matrix :: (Num a, Enum a) => a -> a -> [[a]]
matrix x y =
  [ [ j + i * x
      | j <- [1 .. x]
    ]
    | i <- [0 .. (y -1)]
  ]

vec_mul :: Num a => [a] -> [a] -> [a]
vec_mul x y = [(x !! c) * (y !! c) | c <- [0 .. (length x) -1]]

vec_minus :: Num a => [a] -> [a] -> [a]
vec_minus x y = [(x !! c) - (y !! c) | c <- [0 .. (length x) -1]]

mat_mul :: Num a => [[a]] -> [[a]] -> [[a]]
mat_mul x y = [[sum (vec_mul a ([k !! b | k <- y])) | b <- [0 .. (length (y !! 0)) -1]] | a <- x]

weights :: [[Double]]
weights =
  [ [-0.01, 0.02, 0.03],
    [0.04, -0.05, 0.02],
    [-0.07, 0.03, 0.01],
    [0.05, 0.02, -0.04]
  ]

targets :: [Double]
targets = [1.0, 0.0, 1.0]

output :: Floating a => [[a]] -> [[a]] -> [a]
output x y = [tanh i | i <- ((mat_mul x y) !! 0)]

output' :: Floating a => [[a]] -> [[a]] -> [a]
output' x y = [tanh' i | i <- ((mat_mul x y) !! 0)]

output_s :: [[Double]] -> [[Double]] -> [Double]
output_s x y = [sigmoid i | i <- ((mat_mul x y) !! 0)]

output_s' :: [[Double]] -> [[Double]] -> [Double]
output_s' x y = [sigmoid' i | i <- ((mat_mul x y) !! 0)]

e :: Double
e = 1 + sum [(1 / product [1 .. n]) | n <- [1 .. 10]]

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + e ** (- x))

sigmoid' :: Double -> Double
sigmoid' x = (sigmoid x) * (1 - (sigmoid x))

sigmoid_layer :: [Double] -> [Double]
sigmoid_layer x = [sigmoid (x !! n) | n <- [0 .. (length x) -1]]

sigmoid'_layer :: [Double] -> [Double]
sigmoid'_layer x = [sigmoid' (x !! n) | n <- [0 .. (length x) -1]]

tanh' :: Floating a => a -> a
tanh' x = 1 - ((tanh x) ** 2)

tanh_layer :: Floating a => [a] -> [a]
tanh_layer x = [tanh (x !! n) | n <- [0 .. (length x) -1]]

tanh'_layer :: Floating a => [a] -> [a]
tanh'_layer x = [tanh' (x !! n) | n <- [0 .. (length x) -1]]

err :: Floating a => a -> a -> a
err p t = ((p - t) ** 2) / 2

err_layer :: Floating a => [a] -> [a] -> [a]
err_layer p t = [err (p !! n) (t !! n) | n <- [0 .. (length p) -1]]

err' :: Num a => a -> a -> a
err' p t = p - t

err'_layer :: Num a => [a] -> [a] -> [a]
err'_layer p t = [err' (p !! n) (t !! n) | n <- [0 .. (length p) -1]]

backprop :: [[Double]] -> [[Double]] -> [Double] -> [[Double]]
backprop inpts wgts tgts = [[i * j * 0.05 | j <- (vec_mul (err'_layer (output inpts wgts) tgts) (output' inpts wgts))] | i <- (inpts !! 0)]

update :: [[Double]] -> [[Double]] -> [Double] -> [[Double]]
update inpts wgts tgts = [vec_minus (wgts !! n) ((backprop inpts wgts tgts) !! n) | n <- [0 .. (length wgts) -1]]

updateLoop :: Integer -> [[Double]] -> [[Double]]
updateLoop 0 x = weights
updateLoop z x = update x (updateLoop (z - 1) x) targets

fit :: [[Double]] -> [Double] -> Integer -> [Double]
fit input targets epochs = output input (updateLoop epochs input)
