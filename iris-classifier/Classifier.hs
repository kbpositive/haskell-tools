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
    update,
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
vecByMat x y = [sum ([((y !! d) !! c) * (x !! c) | c <- [0 .. length y - 1]]) | d <- [0 .. length (head y) - 1]]

vecMinus :: [Double] -> [Double] -> [Double]
vecMinus x y = [(x !! c) - (y !! c) | c <- [0 .. length x -1]]

vecMul :: Double -> [Double] -> [Double]
vecMul x y = [x * (y !! c) | c <- [0 .. length y -1]]

matMinus :: [[Double]] -> [[Double]] -> [[Double]]
matMinus x y = [[((x !! c) !! d) - ((y !! c) !! d) | c <- [0 .. length (head x) -1]] | d <- [0 .. length x - 1]]

hadamard :: [Double] -> [Double] -> [Double]
hadamard x y = [(x !! c) * (y !! c) | c <- [0 .. length x -1]]

tensor :: [Double] -> [Double] -> [[Double]]
tensor x y = [[a * b | b <- y] | a <- x]

transpose :: [[Double]] -> [[Double]]
transpose ([] : _) = []
transpose x = map head x : transpose (map tail x)

weights :: [[[Double]]]
weights =
  [ [ [0.01, 0.02, 0.03],
      [0.04, 0.05, 0.02],
      [0.07, 0.03, 0.01],
      [0.02, 0.06, 0.07],
      [0.05, 0.02, 0.04]
    ],
    [ [0.09, 0.02, 0.03],
      [0.04, 0.07, 0.07],
      [0.01, 0.04, 0.08],
      [0.02, 0.02, 0.07],
      [0.02, 0.01, 0.09]
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

output' :: [Double] -> [Double] -> [Double]
output' x y = [sigmoid' i | i <- vecByMat x y]

err :: Double -> Double -> Double
err p t = ((p - t) ** 2) / 2

errLayer :: [Double] -> [Double] -> [Double]
errLayer p t = [err (p !! n) (t !! n) | n <- [0 .. length p -1]]

errLayer' :: [Double] -> [Double] -> [Double] -> [Double]
errLayer' p q t = hadamard (vecMinus p t) q

midLayer' :: [Double] -> [Double] -> [[Double]] -> [Double]
midLayer' p q w = hadamard (vecByMat p (transpose w)) q

update :: [Double] -> [[Double]] -> [Double] -> [[Double]]
update i wgts u = matMinus wgts (tensor i (vecMul 0.1 u))
