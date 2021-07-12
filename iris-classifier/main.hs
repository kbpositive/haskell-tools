module Main where

import Classifier

main :: IO ()
main = do
  raw_data <- getContents
  let line_data = [head raw_data, tail (raw_data) !! 0]
  print (line_data)
