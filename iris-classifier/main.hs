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
  let example = 20

  let o_1 = [1 : output (inp !! n) (weights !! 0) | n <- [0 .. length inp -1]]
  let measure_0 = [errLayer ((feedForward 1 inp weights) !! n) (targets !! n) | n <- [0 .. length inp - 1]]

  let u_0 = [hadamard (vecMinus ((feedForward 1 inp weights) !! n) (targets !! n)) (output' (o_1 !! n) (weights !! 1)) | n <- [0 .. length inp - 1]]
  let u_1 = [hadamard (vecByMat (u_0 !! n) (transpose (weights !! 1))) (1 : (output' (inp !! n) (weights !! 0))) | n <- [0 .. length inp - 1]]
  let updates_0 = [updates (length inp -1) inp u_1, updates (length inp -1) o_1 u_0]
  let weights_1 = [matMinus (weights !! n) (updates_0 !! n) | n <- [0 .. length updates_0 - 1]]

  let o_3 = [1 : output (inp !! n) (weights_1 !! 0) | n <- [0 .. length inp -1]]
  let o_4 = [output n (weights_1 !! 1) | n <- o_3]
  let measure_1 = [errLayer (o_4 !! n) (targets !! n) | n <- [0 .. length inp - 1]]

  let u_2 = [hadamard (vecMinus (o_4 !! n) (targets !! n)) (output' (o_3 !! n) (weights_1 !! 1)) | n <- [0 .. length inp - 1]]
  let u_3 = [hadamard (vecByMat (u_2 !! n) (transpose (weights_1 !! 1))) (1 : (output' (inp !! n) (weights_1 !! 0))) | n <- [0 .. length inp - 1]]
  let updates_1 = [updates (length inp -1) inp u_3, updates (length inp -1) o_3 u_2]
  let weights_2 = [matMinus (weights_1 !! n) (updates_1 !! n) | n <- [0 .. length updates_1 - 1]]

  let o_5 = [1 : output (inp !! n) (weights_2 !! 0) | n <- [0 .. length inp -1]]
  let o_6 = [output n (weights_2 !! 1) | n <- o_5]
  let measure_2 = [errLayer (o_6 !! n) (targets !! n) | n <- [0 .. length inp - 1]]

  print (measure_0 !! example, measure_2 !! example)
