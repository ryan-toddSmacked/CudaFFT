# CudaFFT
C++ namespace wrapper for 1D/2D fft functionality provided by CUDA Toolkit

## Provided Functions

### fft(in, out, N)

- 1D FFT along contiguous memory on device.
- **N**: Number of elements in the pointers

### ifft(in, out, N)

- 1D IFFT along contiguous memory on device. After the inverse transform, the data is scaled. The following should be true; **ifft(fft(x)) == x**
- **N**: Number of elements in the pointers.

### fft(in, out, width, height, dim, pitch_in, pitch_out)

- 1D FFT along the specified dimension of the 2D data. Can perform height 1D FFTs along the width, or width 1D FFTs along the height.
- **width**: Number of elements in the fastest dimension of the 2D data, if column major ordering, this would be along the columns i.e the number of rows.
- **height**: Number of elements in the slowest dimension of the 2D data.
- **dim**: 0 to perform height FFTS along the width of the data, 1 to perform width FFTS along the height of the data
- **pitch_in**: If the input data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.
- **pitch_out**: If the output data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.

### ifft(in, out, width, height, dim, pitch_in, pitch_out)

- 1D IFFT along the specified dimension of the 2D data. Can perform height 1D IFFTs along the width, or width 1D IFFTs along the height. The following should be true; **ifft(fft(x)) == x**
- **width**: Number of elements in the fastest dimension of the 2D data, if column major ordering, this would be along the columns i.e the number of rows.
- **height**: Number of elements in the slowest dimension of the 2D data.
- **dim**: 0 to perform height FFTS along the width of the data, 1 to perform width FFTS along the height of the data
- **pitch_in**: If the input data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.
- **pitch_out**: If the output data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.


### fft2(in, out, width, height, pitch_in, pitch_out)

- 2D FFT.
- **width**: Number of elements in the fastest dimension of the 2D data, if column major ordering, this would be along the columns i.e the number of rows.
- **height**: Number of elements in the slowest dimension of the 2D data.
- **pitch_in**: If the input data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.
- **pitch_out**: If the output data is pitched in memory, this should be used to indicate how many pitched elements there are. 0 indicates no pitch.


