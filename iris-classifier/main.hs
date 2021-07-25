module Main where

import Classifier

main :: IO ()
main =
  do
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
    let epochs = 2
    let example = 30
    let learningRate = 0.01

    let inputData = [inp !! n | n <- [(i `mod` 3) * 50 + (i `div` 3) | i <- [0 .. length inp - 1]]]
    let targetData = [targets !! n | n <- [(i `mod` 3) * 50 + (i `div` 3) | i <- [0 .. length inp - 1]]]

    let measure_0 = measure inputData weights targetData

    let weights_0 = updateLoop learningRate epochs inputData weights targetData

    let measure_1 = measure inputData weights_0 targetData

    print (measure_0 !! example)
    print (measure_1 !! example)

--print (targetData !! example, (feedForward 1 inputData weights) !! example, (feedForward 1 inputData weights_0) !! example)
