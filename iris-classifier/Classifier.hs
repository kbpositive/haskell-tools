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
    feedForward,
    measure,
    outDiffs,
    updateLoop,
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
    [ [0.09, 0.02, 0.08, 0.03, 0.02, 0.08, 0.03, 0.02, 0.08, 0.03, 0.02, 0.08, 0.03, 0.02, 0.08, 0.03],
      [0.04, 0.02, 0.03, 0.05, 0.02, 0.03, 0.05, 0.02, 0.03, 0.05, 0.02, 0.03, 0.05, 0.02, 0.03, 0.05],
      [0.01, 0.01, 0.04, 0.08, 0.01, 0.04, 0.08, 0.01, 0.04, 0.08, 0.01, 0.04, 0.08, 0.01, 0.04, 0.08],
      [0.04, 0.08, 0.02, 0.08, 0.08, 0.02, 0.08, 0.08, 0.02, 0.08, 0.08, 0.02, 0.08, 0.08, 0.02, 0.08],
      [0.02, 0.07, 0.06, 0.03, 0.07, 0.06, 0.03, 0.07, 0.06, 0.03, 0.07, 0.06, 0.03, 0.07, 0.06, 0.03]
    ],
    [ [0.09, 0.02, 0.03],
      [0.06, 0.04, 0.08],
      [0.04, 0.01, 0.07],
      [0.07, 0.04, 0.03],
      [0.06, 0.04, 0.08],
      [0.04, 0.01, 0.07],
      [0.07, 0.04, 0.03],
      [0.06, 0.04, 0.08],
      [0.04, 0.01, 0.07],
      [0.07, 0.04, 0.03],
      [0.06, 0.04, 0.08],
      [0.04, 0.01, 0.07],
      [0.07, 0.04, 0.03],
      [0.06, 0.04, 0.08],
      [0.04, 0.01, 0.07],
      [0.07, 0.04, 0.03],
      [0.09, 0.07, 0.07]
    ]
  ]

e :: Double
e = 1 + sum [1 / product [1 .. n] | n <- [1 .. 10]]

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
errLayer' p q t = hadamard (errLayer p t) q

midLayer' :: [Double] -> [[Double]] -> [Double] -> [Double]
midLayer' p w q = hadamard (vecByMat p (transpose w)) q

updates :: Double -> Int -> [[Double]] -> [[Double]] -> [[Double]]
updates l a i u = if a == 0 then (tensor (i !! a) (vecMul l (u !! a))) else matAdd (updates l (a - 1) i u) (tensor (i !! a) (vecMul l (u !! a)))

feedForward :: Int -> [[Double]] -> [[[Double]]] -> [[Double]]
feedForward a inp wgts =
  if a == 0
    then [1 : output (inp !! n) (wgts !! a) | n <- [0 .. length inp -1]]
    else
      if a < length wgts - 1
        then [1 : output n (wgts !! a) | n <- (feedForward (a -1) inp wgts)]
        else [output n (wgts !! a) | n <- (feedForward (a -1) inp wgts)]

measure :: [[Double]] -> [[[Double]]] -> [[Double]] -> [[Double]]
measure inp wgts tgts = [errLayer ((feedForward (length wgts - 1) inp wgts) !! n) (tgts !! n) | n <- [0 .. length tgts - 1]]

outDiffs :: Int -> [[Double]] -> [[[Double]]] -> [[Double]] -> [[Double]]
outDiffs a inp wgts tgts =
  if a == 0
    then [errLayer' ((feedForward (length wgts - 1) inp wgts) !! n) (tgts !! n) (output' ((feedForward (length wgts - 2) inp wgts) !! n) (wgts !! ((length wgts - 1) - a))) | n <- [0 .. length inp - 1]]
    else
      if a < length wgts - 1
        then [midLayer' ((outDiffs (a - 1) inp wgts tgts) !! n) (wgts !! (length wgts - a)) (1 : (output' ((feedForward ((length wgts - 1) - a) inp wgts) !! n) (wgts !! ((length wgts - 1) - a)))) | n <- [0 .. length inp - 1]]
        else [midLayer' ((outDiffs (a - 1) inp wgts tgts) !! n) (wgts !! (length wgts - a)) (1 : (output' (inp !! n) (wgts !! ((length wgts - 1) - a)))) | n <- [0 .. length inp - 1]]

weightUpdate :: Int -> [[Double]] -> [[[Double]]] -> [[Double]] -> Double -> [[Double]]
weightUpdate a i wgts t l =
  if a == 0
    then matMinus (wgts !! a) (updates l (length i - 1) i (outDiffs (length weights - 1) i wgts t))
    else matMinus (wgts !! a) (updates l (length i - 1) (feedForward (a -1) i wgts) (outDiffs ((length weights - 1) - a) i wgts t))

updateLoop :: Double -> Int -> [[Double]] -> [[[Double]]] -> [[Double]] -> [[[Double]]]
updateLoop learningRate a inp wgts tgts = if a == 0 then [weightUpdate n inp wgts tgts learningRate | n <- [0 .. length weights - 1]] else [weightUpdate n inp (updateLoop learningRate (a -1) inp wgts tgts) tgts learningRate | n <- [0 .. length weights - 1]]
