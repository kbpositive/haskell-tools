module Classifier
  ( matrix,
    vecByMat,
    vecMinus,
    weights,
    e,
    output,
    output',
    err,
    errLayer,
    errLayer',
    sigmoid,
    sigmoid',
    updates,
    weightUpdate,
    hadamard,
    tensor,
    midLayer',
    transpose,
    vecMul,
    matMinus,
    matAdd,
    learningRate,
    feedForward,
    measure,
  )
where

matrix :: (Num a, Enum a) => a -> a -> [[a]]
matrix row col =
  [ [ j + i * col
      | j <- [1 .. col]
    ]
    | i <- [0 .. (row -1)]
  ]

vecByMat :: [Double] -> [[Double]] -> [Double]
vecByMat x y = [sum ([((y !! c) !! d) * (x !! c) | c <- [0 .. length y - 1]]) | d <- [0 .. length (head y) - 1]]

vecMinus :: [Double] -> [Double] -> [Double]
vecMinus x y = [(x !! c) - (y !! c) | c <- [0 .. length x -1]]

vecMul :: Double -> [Double] -> [Double]
vecMul x y = [x * (y !! c) | c <- [0 .. length y -1]]

matMinus :: [[Double]] -> [[Double]] -> [[Double]]
matMinus x y = [[((x !! d) !! c) - ((y !! d) !! c) | c <- [0 .. length (head x) - 1]] | d <- [0 .. length x - 1]]

matAdd :: [[Double]] -> [[Double]] -> [[Double]]
matAdd x y = [[((x !! d) !! c) + ((y !! d) !! c) | c <- [0 .. length (head x) - 1]] | d <- [0 .. length x - 1]]

hadamard :: [Double] -> [Double] -> [Double]
hadamard x y = [(x !! c) * (y !! c) | c <- [0 .. length x -1]]

tensor :: [Double] -> [Double] -> [[Double]]
tensor x y = [[a * b | b <- y] | a <- x]

transpose :: [[Double]] -> [[Double]]
transpose ([] : _) = []
transpose x = map head x : transpose (map tail x)

weights :: [[[Double]]]
weights =
  [ [ [0.01, 0.02, 0.03, 0.07],
      [0.04, 0.05, 0.02, 0.04],
      [0.07, 0.03, 0.01, 0.03],
      [0.02, 0.06, 0.07, 0.08],
      [0.05, 0.02, 0.04, 0.01]
    ],
    [ [0.09, 0.02, 0.03],
      [0.04, 0.07, 0.07],
      [0.01, 0.04, 0.08],
      [0.02, 0.02, 0.07]
    ]
  ]

e :: Double
e = 1 + sum [1 / product [1 .. n] | n <- [1 .. 10]]

learningRate :: Double
learningRate = 0.001

layers :: Int
layers = 2

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + e ** (- x))

sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

output :: [Double] -> [[Double]] -> [Double]
output x y = [sigmoid i | i <- vecByMat x y]

output' :: [Double] -> [[Double]] -> [Double]
output' x y = [sigmoid' i | i <- vecByMat x y]

err :: Double -> Double -> Double
err p t = ((p - t) ** 2) / 2

errLayer :: [Double] -> [Double] -> [Double]
errLayer p t = [err (p !! n) (t !! n) | n <- [0 .. length p -1]]

errLayer' :: [Double] -> [Double] -> [Double] -> [Double]
errLayer' p q t = hadamard (vecMinus p t) q

midLayer' :: [Double] -> [Double] -> [[Double]] -> [Double]
midLayer' p q w = hadamard (vecByMat p (transpose w)) q

updates :: Int -> [[Double]] -> [[Double]] -> [[Double]]
updates a i u = if a == 0 then (tensor (i !! a) (vecMul learningRate (u !! a))) else matAdd (updates (a - 1) i u) (tensor (i !! a) (vecMul learningRate (u !! a)))

weightUpdate :: [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
weightUpdate wgts i u = matMinus wgts (updates (length i - 1) i u)

feedForward :: Int -> [[Double]] -> [[[Double]]] -> [[Double]]
feedForward a inp wgts =
  if a == 0
    then [1 : output (inp !! n) (wgts !! 0) | n <- [0 .. length inp -1]]
    else [output n (wgts !! 1) | n <- (feedForward (a -1) inp wgts)]

measure :: [[Double]] -> [[[Double]]] -> [[Double]] -> [[Double]]
measure inp wgts tgts = [errLayer ((feedForward (length tgts - 1) inp wgts) !! n) (tgts !! n) | n <- [0 .. length tgts - 1]]

--halfProduct :: Int -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
--halfProduct a inp out wgts tgts = if a == 0 then [hadamard (vecMinus ((out!!1) !! n) (tgts !! n)) (output' ((out!!0) !! n) (wgts !! 1)) | n <- [0 .. length inp - 1]]
--else [hadamard (vecByMat ((halfProduct (a-1) inp out wgts tgts) !! n) (transpose (wgts !! 1))) (1 : (output' (inp !! n) (wgts !! 0))) | n <- [0 .. length inp - 1]]
--
--ediffs :: Int -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
--ediffs a inp out wgts tgts = [n | n <- (halfProduct a inp out wgts tgts)]
--
--  let updates_0 = [updates (length inp -1) inp (ediffs layers inp)!!1, updates (length inp -1) o_1 u_0]
--  let weights_1 = [matMinus (weights !! n) (updates_0 !! n) | n <- [0 .. length updates_0 - 1]]
--
--  let o_3 = [1 : output (inp !! n) (weights_1 !! 0) | n <- [0 .. length inp -1]]
--  let o_4 = [output n (weights_1 !! 1) | n <- o_3]
--  let measure_1 = [errLayer (o_4 !! n) (targets !! n) | n <- [0 .. length inp - 1]]
