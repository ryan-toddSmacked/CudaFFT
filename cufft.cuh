#pragma once
#ifndef __CUFFT_CUH__
#define __CUFFT_CUH__

#include <cuComplex.h>
#include <cufft.h>

namespace cufft
{
	// 1D FFT
	// d_in: device input data contiguously stored in device memory
	// d_out: device output data contiguously stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N);


	// 1D FFT
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);


	// Perform 1D FFT on a 2D array along a specified dimension
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the FFT along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch=0, int out_pitch=0);


	// Perform 1D FFT on a 2D array along a specified dimension
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the FFT along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch = 0, int out_pitch = 0);

	// 2D FFT, does work on pitched memory
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch=0, int out_pitch=0);


	// 2D FFT, does work on pitched memory
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fft2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch = 0, int out_pitch = 0);


	// 1D IFFT
	// d_in: device input data contiguously stored in device memory
	// d_out: device output data contiguously stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N);


	// 1D IFFT
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);


	// Perform 1D IFFT on a 2D array along a specified dimension
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the IFFT along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch = 0, int out_pitch = 0);


	// Perform 1D IFFT on a 2D array along a specified dimension
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the IFFT along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch = 0, int out_pitch = 0);


	// 2D IFFT, does work on pitched memory
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch = 0, int out_pitch = 0);


	// 2D IFFT, does work on pitched memory
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t ifft2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch = 0, int out_pitch = 0);


	// 1D fftshift, undefined behavior for overlapping input and output data
	// d_in: device input data contiguously stored in device memory
	// d_out: device output data contiguously stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N);


	// 1D fftshift, undefined behavior for overlapping input and output data
	// d_in: device input data contiguously stored in device memory
	// d_out: device output data contiguously stored in device memory
	// N: number of elements in the input data
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N);


	// Perform 1D fftshift on a 2D array along a specified dimension, undefined behavior for overlapping input and output data
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the fftshift along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch = 0, int out_pitch = 0);


	// Perform 1D fftshift on a 2D array along a specified dimension, undefined behavior for overlapping input and output data
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// dim: dimension to perform the fftshift along. 0 for width, 1 for height
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch = 0, int out_pitch = 0);


	// 2D fftshift, undefined behavior for overlapping input and output data
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch = 0, int out_pitch = 0);


	// 2D fftshift, undefined behavior for overlapping input and output data
	// d_in: device input data stored in device memory
	// d_out: device output data stored in device memory
	// width: width of the input data
	// height: height of the input data
	// in_pitch: pitch of the input data in ELEMENTS. If 0, in_pitch = width
	// out_pitch: pitch of the output data in ELEMENTS. If 0, out_pitch = width
	// cufftResult_t: CUFFT_SUCCESS if success, other values if error
	cufftResult_t fftshift2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch = 0, int out_pitch = 0);

};



#endif // !__CUFFT_CUH__

