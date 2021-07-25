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
  [ [ [0.1, 0.2, 0.3, 0.7],
      [0.4, 0.5, 0.2, 0.4],
      [0.7, 0.3, 0.1, 0.3],
      [0.2, 0.6, 0.7, 0.8],
      [0.5, 0.2, 0.4, 0.1]
    ],
    [ [0.9, 0.2, 0.3],
      [0.4, 0.7, 0.7],
      [0.1, 0.4, 0.8],
      [0.2, 0.2, 0.7]
    ]
  ]

e :: Double
e = 1 + sum [1 / product [1 .. n] | n <- [1 .. 10]]

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

updates :: Double -> Int -> [[Double]] -> [[Double]] -> [[Double]]
updates l a i u = if a == 0 then (tensor (i !! a) (vecMul l (u !! a))) else matAdd (updates (a - 1) i u) (tensor (i !! a) (vecMul l (u !! a)))

feedForward :: Int -> [[Double]] -> [[[Double]]] -> [[Double]]
feedForward a inp wgts =
  if a == 0
    then [1 : output (inp !! n) (wgts !! 0) | n <- [0 .. length inp -1]]
    else [output n (wgts !! 1) | n <- (feedForward (a -1) inp wgts)]

measure :: [[Double]] -> [[[Double]]] -> [[Double]] -> [[Double]]
measure inp wgts tgts = [errLayer ((feedForward 1 inp wgts) !! n) (tgts !! n) | n <- [0 .. length tgts - 1]]

outDiffs :: Int -> [[Double]] -> [[[Double]]] -> [[Double]] -> [[Double]]
outDiffs a inp wgts tgts =
  if a == 0
    then [hadamard (vecMinus ((feedForward 1 inp wgts) !! n) (tgts !! n)) (output' ((feedForward 0 inp wgts) !! n) (wgts !! 1)) | n <- [0 .. length inp - 1]]
    else [hadamard (vecByMat ((outDiffs (a -1) inp wgts tgts) !! n) (transpose (wgts !! 1))) (1 : (output' (inp !! n) (wgts !! 0))) | n <- [0 .. length inp - 1]]

weightUpdate :: [[Double]] -> [[[Double]]] -> [[Double]] -> Double -> [[[Double]]]
weightUpdate i wgts t l =
  [ matMinus (wgts !! 0) (updates l (length i -1) i (outDiffs 1 i wgts t)),
    matMinus (wgts !! 1) (updates l (length i -1) (feedForward 0 i wgts) (outDiffs 0 i wgts t))
  ]

updateLoop :: Double -> Int -> [[Double]] -> [[[Double]]] -> [[Double]] -> [[[Double]]]
updateLoop learningRate a inp wgts tgts = if a == 0 then weightUpdate inp wgts tgts learningRate else weightUpdate inp (updateLoop (a -1) inp wgts tgts) tgts learningRate
