#ifndef DEVICE_ARRAY_CUH__
#define DEVICE_ARRAY_CUH__

#include "SafeCall.h"

#include <vector>
#include <atomic>

template<class T> struct PtrSz {

	__device__ inline T & operator[](int x) const;

	__device__ inline operator T*() const;

	__device__ inline int locate(T element) const;

	T * data;

	size_t size;
};

template<class T> struct PtrStep {

	__device__ inline T * ptr(int y = 0) const;

	T * data;

	size_t step;
};

template<class T> struct PtrStepSz {

	__device__ inline T * ptr(int y = 0) const;

	T * data;

	int cols;

	int rows;

	size_t step;
};

template<class T> struct DeviceArray {

	DeviceArray();

	~DeviceArray();

	DeviceArray(size_t size_);

	DeviceArray(const std::vector<T> & vec);

	void create(size_t size_);

	void upload(const void * data_);

	void upload(const std::vector<T> & vec);

	void upload(const void * data_, size_t size_);

	void download(void * data_) const;

	void download(std::vector<T> & vec) const;

	void download(void * data_, size_t size_) const;

	void clear();

	void release();

	void copyTo(DeviceArray<T> & other) const;

	DeviceArray<T> & operator=(const DeviceArray<T> & other);

	operator T*() const;

	operator PtrSz<T>() const;

	void * data;

	size_t size;

	std::atomic<int> * ref;
};

template<class T> struct DeviceArray2D {

	DeviceArray2D();

	~DeviceArray2D();

	DeviceArray2D(int cols_, int rows_);

	void create(int cols_, int rows_);

	void upload(const void * data_);

	void upload(const void * data_, size_t step_);

	void upload(const void * data_, size_t step_, int cols_, int rows_);

	void download(void * data_, size_t step_) const;

	void clear();

	void release();

	void swap(DeviceArray2D<T> & other);

	void copyTo(DeviceArray2D<T> & other) const;

	DeviceArray2D<T> & operator=(const DeviceArray2D<T> & other);

	operator T*() const;

	operator PtrStep<T>() const;

	operator PtrStepSz<T>() const;

	void * data;

	size_t step;

	int cols, rows;

	std::atomic<int> * ref;
};

//------------------------------------------------------------------
// PtrSz
//------------------------------------------------------------------
template<class T> __device__ inline T & PtrSz<T>::operator [](int x) const {
	return data[x];
}

template<class T> __device__ inline PtrSz<T>::operator T*() const {
	return data;
}

template<class T> __device__ inline int PtrSz<T>::locate(T element) const{
	for(int i=0; i<size; i++){
		if(data[i] == element) {
			return i;
		}
	}
	return -1;
}

//------------------------------------------------------------------
// PtrStep
//------------------------------------------------------------------
template<class T> __device__ inline T * PtrStep<T>::ptr(int y) const {
	return (T*) ((char*) data + y * step);
}

//------------------------------------------------------------------
// PtrStepSz
//------------------------------------------------------------------
template<class T> __device__ inline T * PtrStepSz<T>::ptr(int y) const {
	return (T*) ((char*) data + y * step);
}

//------------------------------------------------------------------
// DeviceArray
//------------------------------------------------------------------
template<class T> DeviceArray<T>::DeviceArray() :
		data(0), ref(0), size(0) {
}

template<class T> DeviceArray<T>::DeviceArray(size_t size_) :
		data(0), ref(0), size(size_) {
	create(size_);
}

template<class T> DeviceArray<T>::DeviceArray(const std::vector<T> & vec) :
		data(0), ref(0), size(vec.size()) {
	create(size);
	upload(vec);
}

template<class T> DeviceArray<T>::~DeviceArray() {
	release();
}

template<class T> void DeviceArray<T>::create(size_t size_) {
	if (data) release();
	SafeCall(cudaMalloc(&data, sizeof(T) * size_));
	size = size_;
	ref = new std::atomic<int>(1);
}

template<class T> void DeviceArray<T>::upload(const void * data_) {
	upload(data_, size);
}

template<class T> void DeviceArray<T>::upload(const std::vector<T> & vec) {
	upload(vec.data(), vec.size());
}

template<class T> void DeviceArray<T>::upload(const void * data_, size_t size_) {
	if (size_ > size) return;
	SafeCall(cudaMemcpy(data, data_, sizeof(T) * size_, cudaMemcpyHostToDevice));
}

template<class T> void DeviceArray<T>::download(void * data_) const {
	download(data_, size);
}

template<class T> void DeviceArray<T>::download(std::vector<T> & vec) const {
	if(vec.size() != size)
		vec.resize(size);
	download((void*) vec.data(), vec.size());
}

template<class T> void DeviceArray<T>::download(void * data_, size_t size_) const {
	SafeCall(cudaMemcpy(data_, data, sizeof(T) * size_,	cudaMemcpyDeviceToHost));
}

template<class T> void DeviceArray<T>::clear() {
	SafeCall(cudaMemset(data, 0, sizeof(T) * size));
}

template<class T> void DeviceArray<T>::release() {
	if (ref && --*ref == 0) {
		delete ref;
		if (data) {
			SafeCall(cudaFree(data));
		}
	}

	size = 0;
	data = 0;
	ref = 0;
}

template<class T> void DeviceArray<T>::copyTo(DeviceArray<T> & other) const {
	if (!data) {
		other.release();
		return;
	}

	other.create(size);
	SafeCall(cudaMemcpy(other.data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template<class T> DeviceArray<T> & DeviceArray<T>::operator=(const DeviceArray<T> & other) {
	if(this != &other) {
		if(other.ref)
			++*other.ref;

		release();

		ref = other.ref;

		size = other.size;

		data = other.data;
	}

	return *this;
}

template<class T> DeviceArray<T>::operator T*() const {
	return (T*)data;
}

template<class T> DeviceArray<T>::operator PtrSz<T>() const {
	PtrSz<T> ps;
	ps.data = (T*) data;
	ps.size = size;
	return ps;
}

//------------------------------------------------------------------
// DeviceArray2D
//------------------------------------------------------------------
template<class T> DeviceArray2D<T>::DeviceArray2D():
		data(0), ref(0), step(0), cols(0), rows(0) {
}

template<class T> DeviceArray2D<T>::DeviceArray2D(int cols_, int rows_):
		data(0), ref(0), step(0), cols(cols_), rows(rows_) {
	create(cols_, rows_);
}

template<class T> DeviceArray2D<T>::~DeviceArray2D() {
	release();
}

template<class T> void DeviceArray2D<T>::create(int cols_, int rows_) {
	if(cols_ > 0 && rows_ > 0) {
		if(data)
			release();

		SafeCall(cudaMallocPitch(&data, &step, sizeof(T) * cols_, rows_));

		cols = cols_;

		rows = rows_;

		ref = new std::atomic<int>(1);
	}
}

template<class T> void DeviceArray2D<T>::upload(const void * data_) {
	upload(data_, sizeof(T) * cols, cols, rows);
}

template<class T> void DeviceArray2D<T>::upload(const void * data_, size_t step_) {
	upload(data_, step_, cols, rows);
}

template<class T> void DeviceArray2D<T>::upload(const void * data_, size_t step_, int cols_, int rows_) {
	if(!data)
		create(cols_, rows_);

	SafeCall(cudaMemcpy2D(data, step, data_, step_, sizeof(T) * cols_, rows_, cudaMemcpyHostToDevice));
}

template<class T> void DeviceArray2D<T>::swap(DeviceArray2D<T> & other) {

	std::swap(ref, other.ref);

	std::swap(data, other.data);

	std::swap(cols, other.cols);

	std::swap(rows, other.rows);

	std::swap(step, other.step);
}

template<class T> void DeviceArray2D<T>::clear() {
	SafeCall(cudaMemset2D(data, step, 0, sizeof(T) * cols, rows));
}

template<class T> void DeviceArray2D<T>::download(void * data_, size_t step_) const {
	if(!data)
		return;
	SafeCall(cudaMemcpy2D(data_, step_, data, step, sizeof(T) * cols, rows, cudaMemcpyDeviceToHost));
}

template<class T> void DeviceArray2D<T>::release() {
	if(ref && --*ref == 0) {
		delete ref;
		if(data)
			SafeCall(cudaFree(data));
	}
	cols = rows = step = 0;
	data = ref = 0;
}

template<class T> void DeviceArray2D<T>::copyTo(DeviceArray2D<T> & other) const {
	if(!data)
		other.release();
	other.create(cols, rows);
	SafeCall(cudaMemcpy2D(other.data, other.step, data, step, sizeof(T) * cols, rows, cudaMemcpyDeviceToDevice));
}

template<class T> DeviceArray2D<T>& DeviceArray2D<T>::operator=(const DeviceArray2D<T>& other) {
	if(this != &other) {
		if(other.ref)
			++*other.ref;
		release();

		data = other.data;

		step = other.step;

		cols = other.cols;

		rows = other.rows;

		ref = other.ref;
	}

	return * this;
}

template<class T> DeviceArray2D<T>::operator T*() const {
	return (T*)data;
}

template<class T> DeviceArray2D<T>::operator PtrStep<T>() const {
	PtrStep<T> ps;

	ps.data = (T*)data;
	ps.step = step;

	return ps;
}

template<class T> DeviceArray2D<T>::operator PtrStepSz<T>() const {
	PtrStepSz<T> psz;

	psz.data = (T*)data;
	psz.cols = cols;
	psz.rows = rows;
	psz.step = step;

	return psz;
}

#endif
