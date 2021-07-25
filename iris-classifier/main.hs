module Main where

import Classifier

main :: IO ()
main = do
  raw_data <- getContents
  -- data parsing
  let pos = 0 : [x | (x, y) <- zip [0 ..] raw_data, [y] == "\n"]
  let data_split = [[raw_data !! i | i <- [pos !! i .. (pos !! (i + 1)) -1]] | i <- [0 .. length pos - 2]]
  let iris_data = head data_split : [tail i | i <- tail data_split]
  let commas_pos = [0 : [x + 1 | (x, y) <- zip [0 ..] n, [y] == "," || x == (length n -1)] | n <- iris_data]
  let instances_split = [[[(iris_data !! n) !! i | i <- [(commas_pos !! n) !! i .. ((commas_pos !! n) !! (i + 1)) -1]] | i <- [0 .. length (commas_pos !! n) - 2]] | n <- [0 .. length iris_data -1]]
  let inp = [(1.0 : [read (init n) :: Double | n <- init i]) | i <- instances_split]
  let commas_pos = [0 : [x + 1 | (x, y) <- zip [0 ..] n, [y] == "," || x == (length n -1)] | n <- iris_data]
  let labels = [last i | i <- instances_split]
  let individuals = [labels !! (i - 1) | i <- [1 .. length labels - 1], labels !! (i -1) /= labels !! i || i == (length labels - 1)]
  let target_labels = [[fromIntegral (fromEnum (i == j)) | j <- [0 .. length individuals - 1]] | i <- [0 .. length individuals - 1]]
  let targets = [[fromIntegral (fromEnum (i == n)) | i <- individuals] | n <- labels]

  -- test inputs
  let n = 120

  let o_1 = [1 : output (inp !! n) (weights !! 0) | n <- [0 .. length inp -1]]
  let o_2 = [output n (weights !! 1) | n <- o_1]
  let measure = [errLayer (o_2 !! n) (targets !! n) | n <- [0 .. length inp - 1]]

  let u_0 = [hadamard (vecMinus (o_2 !! n) (targets !! n)) (output' (o_1 !! n) (weights !! 1)) | n <- [0 .. length inp - 1]]
  let u_1 = [hadamard (vecByMat (u_0 !! n) (transpose (weights !! 1))) (1 : (output' (inp !! n) (weights !! 0))) | n <- [0 .. length inp - 1]]

  let newWeights = [update (weights !! 0) (inp !! n) (u_1 !! n), update (weights !! 1) (o_1 !! n) (u_0 !! n)]

  print (o_2 !! n, output (1 : (output (inp !! n) (newWeights !! 0))) (newWeights !! 1), (targets !! n))
