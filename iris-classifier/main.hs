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
  let inp = [[read (init n) :: Double | n <- init i] | i <- instances_split]
  print (fit inp targets 7000)
