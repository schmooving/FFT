# Hellooo
FFT Python Program Project
Hello, this is the program I am currently working for my Special Topics Course. The goal I have set right now is to be able to solve a problem from my textbook, "Numerical Analysis, 8th Edition" written by Richard L. Burden.
The question in question is the following,
Let f(x) = x^4 - 3x^3 + 2x^2 - tanx(x-2). To determine the itronometric interpolating polynomial of degree four for the data {(x_j, y_j)}^7, j=0, where x_j = j/4 and y_j = f(x_j), requires a transformation of the interval [0,2] to [-pi, pi]. This is given by z_j = pi(x_j -1), so that the input data to the algorithm is {z_j, f( 1+(z_j/pi)}^7, j=0. The interpolating polynomial in z is S_4(x) = 0.0761979 + 0.771841cos(z) + 0.0173037cos(2z) + 0.00686304cos(3z) - 0.000578545cos(4z) - 0.386374sin(z) + 0.0468750sin(2z) - 0.0113738sin(3z). The trigonometric polynomial s_4(x) on [0,2] is obtained by subtituting z=pi(x-1) into S_4(z).
From this problem, I attempt to get the f(x) and S(x) coefficient values, with the end goal of the program to solve wave and heat equations.
I have made progress in creating this program, BUT i have come across an issue regarding the values output from the program. Having difficulty pinpointing why this occurs, and am open for ANY help, and would not mind paying for solutions if necessary, just let me know.
import numpy as np
import matplotlib.pyplot as plt
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the function
    def f(x):
    return x**4 - 3*x**3 + 2*x**2 - np.tan(x)*(x-2)
    
    # Number of sample points
    num_points = 8
    
    # Degree of the trigonometric interpolating polynomial
    degree = 4
    
    # Sample the function
    def sample_function(func, num_points):
    x_j = np.linspace(0, 2, num_points)  # Sample points from 0 to 2
    y_j = func(x_j)  # Evaluate the function at these points
    return x_j, y_j
    
    # Transform the interval [0, 2] to [-π, π] using z_j = π(x_j - 1)
    def transform_interval(x_j):
    return np.pi * (x_j - 1)
    
    # Compute the trigonometric interpolating polynomial using FFT
    def trigonometric_interpolation(x_j, y_j, degree):
    n = len(x_j)
    # Transform x_j to z_j
    z_j = transform_interval(x_j)
    
    # Compute FFT of y_j
    fft_coeffs = np.fft.fft(y_j) / n  # Normalize by number of points
    
    # Truncate to the desired degree (highest frequency = degree)
    truncated_coeffs = np.zeros(n, dtype=complex)
    truncated_coeffs[:degree + 1] = fft_coeffs[:degree + 1]  # Keep up to degree
    
    # Inverse FFT to get interpolated values
    interpolated_values = np.fft.ifft(truncated_coeffs * n).real
    
    # Extract coefficients for trigonometric polynomial
    a_0 = fft_coeffs[0].real
    a_coeffs = [2 * fft_coeffs[k].real for k in range(1, degree + 1)]
    b_coeffs = [-2 * fft_coeffs[k].imag for k in range(1, degree + 1)]
    if degree % 2 == 0:
        a_coeffs[-1] = fft_coeffs[degree].real  # Adjust last cosine coefficient
    
    return z_j, interpolated_values, a_0, a_coeffs, b_coeffs
    
    def main():
    func = f
    
    # Sample the function
    x_j, y_j = sample_function(func, num_points)
    
    # Transform and interpolate
    z_j, interpolated_values, a_0, a_coeffs, b_coeffs = trigonometric_interpolation(x_j, y_j, degree)
    
    # Points to evaluate
    x_values = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
    f_values = [func(x) for x in x_values]
    S_4_values = [np.polyval(np.polyfit(x_j, y_j, degree), x) for x in x_values]
    errors = [abs(f_val - S_4_val) for f_val, S_4_val in zip(f_values, S_4_values)]
    
    # Print the values
    print("x values =", x_values)
    print("f(x) values =", f_values)
    print("S_4(x) values =", S_4_values)
    print("abs(f(x) - S_4(x)) =", errors)
    
    # Print the results in table format
    print("\n")
    print("x\tf(x)\t\tS_4(x)\t\tabs(f(x)-S_4(x))")
    for x, f_val, S_4_val, error in zip(x_values, f_values, S_4_values, errors):
        print(f"{x:.3f}\t{f_val:.5f}\t{S_4_val:.5f}\t{error:.5e}")
    
    # Generate fine grid for plotting
    x_plot = np.linspace(0, 2, 10)
    y_f_plot = [func(x) for x in x_plot]
    y_S_4_plot = [np.polyval(np.polyfit(x_j, y_j, degree), x) for x in x_plot]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_f_plot, label='y = f(x)', linestyle='--')
    plt.plot(x_plot, y_S_4_plot, label='y = S_4(x)')
    plt.scatter(x_values, f_values, color='red', label='Data Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function f(x) and Trigonometric Interpolating Polynomial S_4(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if __name__ == "__main__":
    main()
