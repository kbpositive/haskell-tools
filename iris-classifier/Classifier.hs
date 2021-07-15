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
    outputS,
    outputS',
    err,
    err',
    err_layer,
    err'_layer,
    sigmoid,
    sigmoid',
    sigmoidLayer,
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
matrix row col =
  [ [ j + i * col
      | j <- [1 .. col]
    ]
    | i <- [0 .. (row -1)]
  ]

vec_mul :: [Double] -> [Double] -> [Double]
vec_mul x y = [(x !! c) * (y !! c) | c <- [0 .. (length x) -1]]

vec_minus :: [Double] -> [Double] -> [Double]
vec_minus x y = [(x !! c) - (y !! c) | c <- [0 .. (length x) -1]]

mat_mul :: [[Double]] -> [[Double]] -> [[Double]]
mat_mul y x = [[sum (vec_mul a ([k !! b | k <- y])) | b <- [0 .. (length (y !! 0)) -1]] | a <- x]

weights :: [[Double]]
weights =
  [ [-0.01, 0.02, 0.03],
    [0.04, -0.05, 0.02],
    [-0.07, 0.03, 0.01],
    [0.05, 0.02, -0.04]
  ]

targets :: [[Double]]
targets = [[1.0, 0.0, 1.0]]

output :: [[Double]] -> [[Double]] -> [[Double]]
output y x = [[tanh i | i <- ((mat_mul x y) !! n)] | n <- [0 .. length y - 1]]

output' :: [[Double]] -> [[Double]] -> [[Double]]
output' y x = [[tanh' i | i <- ((mat_mul x y) !! n)] | n <- [0 .. length y - 1]]

outputS :: [[Double]] -> [[Double]] -> [[Double]]
outputS y x = [[sigmoid i | i <- ((mat_mul x y) !! n)] | n <- [0 .. length y - 1]]

outputS' :: [[Double]] -> [[Double]] -> [[Double]]
outputS' y x = [[sigmoid' i | i <- ((mat_mul x y) !! n)] | n <- [0 .. length y - 1]]

e :: Double
e = 1 + sum [(1 / product [1 .. n]) | n <- [1 .. 10]]

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + e ** (- x))

sigmoid' :: Double -> Double
sigmoid' x = (sigmoid x) * (1 - (sigmoid x))

sigmoidLayer :: [Double] -> [Double]
sigmoidLayer x = [sigmoid (x !! n) | n <- [0 .. (length x) -1]]

sigmoid'_layer :: [Double] -> [Double]
sigmoid'_layer x = [sigmoid' (x !! n) | n <- [0 .. (length x) -1]]

tanh' :: Floating a => a -> a
tanh' x = 1 - ((tanh x) ** 2)

tanh_layer :: Floating a => [a] -> [a]
tanh_layer x = [tanh (x !! n) | n <- [0 .. (length x) -1]]

tanh'_layer :: Floating a => [a] -> [a]
tanh'_layer x = [tanh' (x !! n) | n <- [0 .. (length x) -1]]

err :: Double -> Double -> Double
err p t = ((p - t) ** 2) / 2

err_layer :: [[Double]] -> [[Double]] -> [[Double]]
err_layer p t = [[err ((p !! n) !! m) ((t !! n) !! m) | m <- [0 .. length (t !! n) - 1]] | n <- [0 .. length p -1]]

err' :: Double -> Double -> Double
err' p t = p - t

err'_layer :: [[Double]] -> [[Double]] -> [[Double]]
err'_layer p t = [[err' ((p !! n) !! m) ((t !! n) !! m) | m <- [0 .. length (p !! n) - 1]] | n <- [0 .. length p -1]]

backprop :: [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
backprop inpts wgts tgts =
  [ [ sum
        ( map (* (((inpts !! n) !! q) * 0.05)) (vec_mul ((err'_layer (output inpts wgts) tgts) !! n) ((output' inpts wgts) !! n))
        )
      | q <- [0 .. length (wgts !! n) - 1]
    ]
    | n <- [0 .. length wgts - 1]
  ]

update :: [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
update inpts wgts tgts = [vec_minus (wgts !! n) ((backprop inpts wgts tgts) !! n) | n <- [0 .. length wgts -1]]

updateLoop :: Integer -> [[Double]] -> [[Double]] -> [[Double]]
updateLoop 0 x y = weights
updateLoop z x y = update x (updateLoop (z - 1) x y) y

fit :: [[Double]] -> [[Double]] -> Integer -> [[Double]]
fit inputs targets epochs = output inputs (updateLoop epochs inputs targets)
