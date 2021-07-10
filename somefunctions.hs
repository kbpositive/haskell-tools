matrix x y = [
    [
        j+i*x |
        j <- [1..x]] |
    i <- [0..(y-1)]]

vec_mul x y = [(x!!c)*(y!!c)| c <- [0..(length x)-1]]

mat_mul x y = [[sum (vec_mul a ([k!!b|k<-y])) | b <- [0..(length x)-1]] | a <- x]