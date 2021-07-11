module Main where

import Classifier

main :: IO ()
main = do
  print (fit input targets 5)
