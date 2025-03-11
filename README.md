# Hellooo
FFT Python Program Project
Hello, this is the program I am currently working for my Special Topics Course. The goal I have set right now is to be able to solve a problem from my textbook, "Numerical Analysis, 8th Edition" written by Richard L. Burden.
The question in question is the following,
Let f(x) = x^4 - 3x^3 + 2x^2 - tanx(x-2). To determine the itronometric interpolating polynomial of degree four for the data {(x_j, y_j)}^7, j=0, where x_j = j/4 and y_j = f(x_j), requires a transformation of the interval [0,2] to [-pi, pi]. This is given by z_j = pi(x_j -1), so that the input data to the algorithm is {z_j, f( 1+(z_j/pi)}^7, j=0. The interpolating polynomial in z is S_4(x) = 0.0761979 + 0.771841cos(z) + 0.0173037cos(2z) + 0.00686304cos(3z) - 0.000578545cos(4z) - 0.386374sin(z) + 0.0468750sin(2z) - 0.0113738sin(3z). The trigonometric polynomial s_4(x) on [0,2] is obtained by subtituting z=pi(x-1) into S_4(z).
From this problem, I attempt to get the f(x) and S(x) coefficient values, with the end goal of the program to solve wave and heat equations.
I have made progress in creating this program, BUT i have come across an issue regarding the values output from the program. Having difficulty pinpointing why this occurs, and am open for ANY help, and would not mind paying for solutions if necessary, just let me know.
