#include "cufft.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdint.h>		// uintptr_t


// ifft2 scale kernel. Divide each element by the number of elements in the array
static __global__ void ifft2_scale_kernel(cuFloatComplex* d_data, int width, int height, double scale, int pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx = 0;
	if (x < width && y < height)
	{
		idx = (size_t)y * pitch + x;
		d_data[idx].x *= scale;
		d_data[idx].y *= scale;
	}
}

// ifft2 scale kernel. Divide each element by the number of elements in the array
static __global__ void ifft2_scale_kernel(cuDoubleComplex* d_data, int width, int height, double scale, int pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx = 0;
	if (x < width && y < height)
	{
		idx = (size_t)y * pitch + x;
		d_data[idx].x *= scale;
		d_data[idx].y *= scale;
	}
}

// ifft scale kernel. Divide each element by the number of elements in the array
static __global__ void ifft_scale_kernel(cuFloatComplex* d_data, int N, double scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		d_data[idx].x *= scale;
		d_data[idx].y *= scale;
	}
}

// ifft scale kernel. Divide each element by the number of elements in the array
static __global__ void ifft_scale_kernel(cuDoubleComplex* d_data, int N, double scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		d_data[idx].x *= scale;
		d_data[idx].y *= scale;
	}
}

// 1D fft shift kernel
static __global__ void fftshift_even_kernel(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		// Pointers are gaurenteed to be different, so no need for a temporary variable
		// N is gaurenteed to be even

		int idx2 = (idx + N / 2) % N;
		d_out[idx] = d_in[idx2];
	}
}

// 1D fft shift kernel
static __global__ void fftshift_odd_kernel(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int half_N = N / 2;
	if (idx < N)
	{
		if (idx <= half_N)
		{
			int idx2 = idx + half_N;
			d_out[idx2] = d_in[idx];
		}
		else
		{
			int idx2 = idx - half_N - 1;
			d_out[idx2] = d_in[idx];
		}
	}
}

// 1D fft shift kernel
static __global__ void fftshift_even_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		// Pointers are gaurenteed to be different, so no need for a temporary variable
		// N is gaurenteed to be even

		int idx2 = (idx + N / 2) % N;
		d_out[idx] = d_in[idx2];
	}
}

// 1D fft shift kernel
static __global__ void fftshift_odd_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int half_N = N / 2;
	if (idx < N)
	{
		if (idx <= half_N)
		{
			int idx2 = idx + half_N;
			d_out[idx2] = d_in[idx];
		}
		else
		{
			int idx2 = idx - half_N - 1;
			d_out[idx2] = d_in[idx];
		}
	}
}

static __global__ void fftshift_even_kernel_width(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the width, y will remain the same

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;

	if (x < width && y < height)
	{
		int x2 = (x + half_width) % width;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y * in_pitch + x2;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift_odd_kernel_width(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the width, y will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			int x2 = x + half_width;
			size_t idx = (size_t)y * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = x - half_width - 1;
			size_t idx = (size_t)y * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift_even_kernel_width(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the width, y will remain the same

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;

	if (x < width && y < height)
	{
		int x2 = (x + half_width) % width;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y * in_pitch + x2;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift_odd_kernel_width(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the width, y will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			int x2 = x + half_width;
			size_t idx = (size_t)y * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = x - half_width - 1;
			size_t idx = (size_t)y * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}


static __global__ void fftshift_even_kernel_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the height, x will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		int y2 = (y + half_height) % height;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y2 * in_pitch + x;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift_odd_kernel_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the height, x will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (y <= half_height)
		{
			int y2 = y + half_height;
			size_t idx = (size_t)y2 * out_pitch + x;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int y2 = y - half_height - 1;
			size_t idx = (size_t)y2 * out_pitch + x;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift_even_kernel_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the height, x will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		int y2 = (y + half_height) % height;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y2 * in_pitch + x;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift_odd_kernel_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Only shifting along the height, x will remain the same
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (y <= half_height)
		{
			int y2 = y + half_height;
			size_t idx = (size_t)y2 * out_pitch + x;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int y2 = y - half_height - 1;
			size_t idx = (size_t)y2 * out_pitch + x;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift2d_even_width_even_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		int x2 = (x + half_width) % width;
		int y2 = (y + half_height) % height;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y2 * in_pitch + x2;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift2d_odd_width_even_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			int x2 = x + half_width;
			int y2 = (y + half_height) % height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = x - half_width - 1;
			int y2 = (y + half_height) % height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift2d_even_width_odd_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (y <= half_height)
		{
			int x2 = (x + half_width) % width;
			int y2 = y + half_height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = (x + half_width) % width;
			int y2 = y - half_height - 1;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift2d_odd_width_odd_height(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			if (y <= half_height)
			{
				int x2 = x + half_width;
				int y2 = y + half_height;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
			else
			{
				int x2 = x + half_width;
				int y2 = y - half_height - 1;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
		}
		else
		{
			if (y <= half_height)
			{
				int x2 = x - half_width - 1;
				int y2 = y + half_height;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
			else
			{
				int x2 = x - half_width - 1;
				int y2 = y - half_height - 1;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
		}
	}
}


static __global__ void fftshift2d_even_width_even_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		int x2 = (x + half_width) % width;
		int y2 = (y + half_height) % height;
		size_t idx = (size_t)y * out_pitch + x;
		size_t idx2 = (size_t)y2 * in_pitch + x2;
		d_out[idx] = d_in[idx2];
	}
}

static __global__ void fftshift2d_odd_width_even_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			int x2 = x + half_width;
			int y2 = (y + half_height) % height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = x - half_width - 1;
			int y2 = (y + half_height) % height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift2d_even_width_odd_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (y <= half_height)
		{
			int x2 = (x + half_width) % width;
			int y2 = y + half_height;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
		else
		{
			int x2 = (x + half_width) % width;
			int y2 = y - half_height - 1;
			size_t idx = (size_t)y2 * out_pitch + x2;
			size_t idx2 = (size_t)y * in_pitch + x;
			d_out[idx] = d_in[idx2];
		}
	}
}

static __global__ void fftshift2d_odd_width_odd_height(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	// Shifting along both the width and height
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int half_width = width / 2;
	int half_height = height / 2;
	if (x < width && y < height)
	{
		if (x <= half_width)
		{
			if (y <= half_height)
			{
				int x2 = x + half_width;
				int y2 = y + half_height;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
			else
			{
				int x2 = x + half_width;
				int y2 = y - half_height - 1;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
		}
		else
		{
			if (y <= half_height)
			{
				int x2 = x - half_width - 1;
				int y2 = y + half_height;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
			else
			{
				int x2 = x - half_width - 1;
				int y2 = y - half_height - 1;
				size_t idx = (size_t)y2 * out_pitch + x2;
				size_t idx2 = (size_t)y * in_pitch + x;
				d_out[idx] = d_in[idx2];
			}
		}
	}
}




cufftResult_t cufft::fft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N)
{
	cufftHandle plan;
	cufftResult result = cufftPlan1d(&plan, N, CUFFT_C2C, 1);

	if (result != CUFFT_SUCCESS)
		return result;
	
	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);
	
	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N)
{
	cufftHandle plan;
	cufftResult result = cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecZ2Z(plan, (cufftDoubleComplex*)d_in, (cufftDoubleComplex*)d_out, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::fft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	// 2D FFT along the dimension specified by dim, either rows or columns

	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;

	int rank = 1;

	if (dim == 0)
	{
		// FFT along the width, the fastest changing dimension in memory
		// There are height rows, each of width elements
		int n[] = { width };
		int inembed[] = { width };
		int istride = 1;
		int idist = in_pitch ? (in_pitch) : (width);
		int onembed[] = { width };
		int ostride = 1;
		int odist = out_pitch ? (out_pitch) : (width);
		int batch = height;		// number of transforms to perform

		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
	}
	else if (dim == 1)
	{
		int n[] = { height };
		int inembed[] = { width };
		int istride = in_pitch ? (in_pitch) : (width);
		int idist = 1;
		int onembed[] = { width };
		int ostride = out_pitch ? (out_pitch) : (width);
		int odist = 1; 
		int batch = width;		// number of transforms to perform

		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
	}
	else
	{
		return CUFFT_SETUP_FAILED;
	}

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS)
	{
		fprintf(stderr, "cufftExecC2C failed\n");
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::fft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	// 2D FFT along the dimension specified by dim, either rows or columns

	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;

	int rank = 1;

	if (dim == 0)
	{
		// FFT along the width, the fastest changing dimension in memory
		// There are height rows, each of width elements
		int n[] = { width };
		int istride = 1, ostride = 1;
		int idist = in_pitch ? (in_pitch) : (width);
		int odist = out_pitch ? (out_pitch) : (width);
		int inembed[] = { in_pitch ? (in_pitch) : (width) };
		int onembed[] = { out_pitch ? (out_pitch) : (width) };
		int batch = height;		// number of transforms to perform

		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
	}
	else if (dim == 1)
	{
		// FFT along the height, the slowest changing dimension in memory, distance between consecutive elements is width
		// There are width columns, each of height elements

		int n[] = { height };
		int istride = in_pitch ? (in_pitch) : (width);
		int ostride = out_pitch ? (out_pitch) : (width);
		int idist = 1, odist = 1;
		int inembed[] = { in_pitch ? (in_pitch) : (width) };
		int onembed[] = { out_pitch ? (out_pitch) : (width) };
		int batch = width;		// number of transforms to perform

		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
	}
	else
	{
		return CUFFT_SETUP_FAILED;
	}

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::fft2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;

	int rank = 2;
	int n[] = { height, width };
	int istride = 1, ostride = 1;
	int idist = 1, odist = 1;
	int inembed[] = { height, in_pitch ? (in_pitch) : (width) };
	int onembed[] = { height, out_pitch ? (out_pitch) : (width) };
	int batch = 1;

	result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::fft2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;

	int rank = 2;
	int n[] = { height, width };
	int istride = 1, ostride = 1;
	int idist = 1, odist = 1;
	int inembed[] = { height, in_pitch ? (in_pitch) : (width) };
	int onembed[] = { height, out_pitch ? (out_pitch) : (width) };
	int batch = 1;

	result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecZ2Z(plan, (cufftDoubleComplex*)d_in, (cufftDoubleComplex*)d_out, CUFFT_FORWARD);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	return cufftDestroy(plan);
}

cufftResult_t cufft::ifft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N)
{
	cufftHandle plan;
	cufftResult result = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);
	if (result != CUFFT_SUCCESS)
		return result;

	double scale = 1.0 / (double)N;

	ifft_scale_kernel << <grid, block >> > (d_out, N, scale);
	cudaError_t err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::ifft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N)
{
	cufftHandle plan;
	cufftResult result = cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecZ2Z(plan, (cufftDoubleComplex*)d_in, (cufftDoubleComplex*)d_out, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);
	if (result != CUFFT_SUCCESS)
		return result;

	double scale = 1.0 / (double)N;
	ifft_scale_kernel << <grid, block >> > (d_out, N, scale);

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::ifft(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;
	cudaError_t err = cudaSuccess;
	int rank = 1;
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	if (dim == 0)
	{
		// FFT along the width, the fastest changing dimension in memory
		// There are height rows, each of width elements
		int n[] = { width };
		int inembed[] = { width };
		int istride = 1;
		int idist = in_pitch ? (in_pitch) : (width);
		int onembed[] = { width };
		int ostride = 1;
		int odist = out_pitch ? (out_pitch) : (width);
		int batch = height;		// number of transforms to perform
		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
	}
	else if (dim == 1)
	{
		int n[] = { height };
		int inembed[] = { width };
		int istride = in_pitch ? (in_pitch) : (width);
		int idist = 1;
		int onembed[] = { width };
		int ostride = out_pitch ? (out_pitch) : (width);
		int odist = 1;
		int batch = width;		// number of transforms to perform
		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
	}
	else
	{
		return CUFFT_SETUP_FAILED;
	}

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);
	if (result != CUFFT_SUCCESS)
		return result;

	double scale = dim == 0 ? 1.0 / (double)width : 1.0 / (double)height;
	//double scale = 1.0;
	ifft2_scale_kernel << <grid, block >> > (d_out, width, height, scale, out_pitch ? (out_pitch) : width);
	err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::ifft(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;
	cudaError_t err = cudaSuccess;
	int rank = 1;
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	if (dim == 0)
	{
		// FFT along the width, the fastest changing dimension in memory
		// There are height rows, each of width elements
		int n[] = { width };
		int inembed[] = { width };
		int istride = 1;
		int idist = in_pitch ? (in_pitch) : (width);
		int onembed[] = { width };
		int ostride = 1;
		int odist = out_pitch ? (out_pitch) : (width);
		int batch = height;		// number of transforms to perform
		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
	}
	else if (dim == 1)
	{
		int n[] = { height };
		int inembed[] = { width };
		int istride = in_pitch ? (in_pitch) : (width);
		int idist = 1;
		int onembed[] = { width };
		int ostride = out_pitch ? (out_pitch) : (width);
		int odist = 1;
		int batch = width;		// number of transforms to perform
		result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
	}
	else
	{
		return CUFFT_SETUP_FAILED;
	}

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecZ2Z(plan, (cuDoubleComplex*)d_in, (cuDoubleComplex*)d_out, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);
	if (result != CUFFT_SUCCESS)
		return result;

	double scale = dim == 0 ? 1.0 / (double)width : 1.0 / (double)height;
	//double scale = 1.0;
	ifft2_scale_kernel << <grid, block >> > (d_out, width, height, scale, out_pitch ? (out_pitch) : width);
	err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::ifft2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;
	cudaError_t err = cudaSuccess;

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	int rank = 2;
	int n[] = { height, width };
	int istride = 1, ostride = 1;
	int idist = 1, odist = 1;
	int inembed[] = { height, in_pitch ? (in_pitch) : (width) };
	int onembed[] = { height, out_pitch ? (out_pitch) : (width) };
	int batch = 1;

	result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);

	if (result != CUFFT_SUCCESS)
		return result;

	double scale = 1.0 / (double)(width * height);

	ifft2_scale_kernel << <grid, block >> > (d_out, width, height, scale, out_pitch ? (out_pitch) : width);
	err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::ifft2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	cufftHandle plan = 0;
	cufftResult result = CUFFT_SUCCESS;
	cudaError_t err = cudaSuccess;

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	int rank = 2;
	int n[] = { height, width };
	int istride = 1, ostride = 1;
	int idist = 1, odist = 1;
	int inembed[] = { height, in_pitch ? (in_pitch) : (width) };
	int onembed[] = { height, out_pitch ? (out_pitch) : (width) };
	int batch = 1;

	result = cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);

	if (result != CUFFT_SUCCESS)
		return result;

	result = cufftExecZ2Z(plan, (cufftDoubleComplex*)d_in, (cufftDoubleComplex*)d_out, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		return result;
	}

	result = cufftDestroy(plan);
	if (result != CUFFT_SUCCESS)
		return result;

	double scale = 1.0 / (double)(width * height);

	ifft2_scale_kernel << <grid, block >> > (d_out, width, height, scale, out_pitch ? (out_pitch) : width);
	err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::fftshift(const cuFloatComplex* d_in, cuFloatComplex* d_out, int N)
{
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + N * sizeof(cuFloatComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	if (N % 2 == 0)
		fftshift_even_kernel << <grid, block >> > (d_in, d_out, N);
	else
		fftshift_odd_kernel << <grid, block >> > (d_in, d_out, N);

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;
	
	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::fftshift(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int N)
{
	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + N * sizeof(cuFloatComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	if (N % 2 == 0)
		fftshift_even_kernel << <grid, block >> > (d_in, d_out, N);
	else
		fftshift_odd_kernel << <grid, block >> > (d_in, d_out, N);

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::fftshift(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + height * in_pitch * sizeof(cuFloatComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	in_pitch = in_pitch ? in_pitch : width;
	out_pitch = out_pitch ? out_pitch : width;

	if (dim == 0)
	{
		if (width % 2 == 0)
			fftshift_even_kernel_width << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift_odd_kernel_width << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else if (dim == 1)
	{
		if (height % 2 == 0)
			fftshift_even_kernel_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift_odd_kernel_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else
	{
		return cufftResult_t::CUFFT_SETUP_FAILED;
	}

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;

}

cufftResult_t cufft::fftshift(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int dim, int in_pitch, int out_pitch)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + height * in_pitch * sizeof(cuDoubleComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	in_pitch = in_pitch ? in_pitch : width;
	out_pitch = out_pitch ? out_pitch : width;

	if (dim == 0)
	{
		if (width % 2 == 0)
			fftshift_even_kernel_width << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift_odd_kernel_width << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else if (dim == 1)
	{
		if (height % 2 == 0)
			fftshift_even_kernel_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift_odd_kernel_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else
	{
		return cufftResult_t::CUFFT_SETUP_FAILED;
	}

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::fftshift2(const cuFloatComplex* d_in, cuFloatComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + height * in_pitch * sizeof(cuFloatComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	in_pitch = in_pitch ? in_pitch : width;
	out_pitch = out_pitch ? out_pitch : width;

	if (width % 2 == 0)
	{
		if (height % 2 == 0)
			fftshift2d_even_width_even_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift2d_even_width_odd_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else
	{
		if (height % 2 == 0)
			fftshift2d_odd_width_even_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift2d_odd_width_odd_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

cufftResult_t cufft::fftshift2(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int width, int height, int in_pitch, int out_pitch)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// Check to see if pointers overlap at all, if they do, just print a warning statement
	// The results are undefined, but the program shouldnt crash, unless N is out of bounds
	uintptr_t in_ptr = (uintptr_t)d_in;
	uintptr_t out_ptr = (uintptr_t)d_out;
	uintptr_t in_end = in_ptr + height * in_pitch * sizeof(cuDoubleComplex);

	if (in_ptr <= out_ptr && out_ptr < in_end)
	{
		fprintf(stderr, "Warning: fftshift: input and output pointers overlap\n");
	}

	in_pitch = in_pitch ? in_pitch : width;
	out_pitch = out_pitch ? out_pitch : width;

	if (width % 2 == 0)
	{
		if (height % 2 == 0)
			fftshift2d_even_width_even_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift2d_even_width_odd_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}
	else
	{
		if (height % 2 == 0)
			fftshift2d_odd_width_even_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
		else
			fftshift2d_odd_width_odd_height << <grid, block >> > (d_in, d_out, width, height, in_pitch, out_pitch);
	}

	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		return cufftResult_t::CUFFT_EXEC_FAILED;

	return cufftResult_t::CUFFT_SUCCESS;
}

