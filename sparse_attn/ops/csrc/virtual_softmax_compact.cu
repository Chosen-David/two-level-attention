// Experimental virtual token-domain softmax, no production dispatch.
// Reduction scheduling follows PyTorch v2.8.0 ATen/native/cuda/SoftMax.cu
// (generic and shared CUDA kernels); BSD-3-Clause upstream license here.
// Changes: gather sparse scores through per-token owners; retain only row stats
// between passes instead of materializing the dense score input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cmath>

namespace {
template<typename T> struct Add {
  __device__ T operator()(T a,T b) const {return a+b;}
  __device__ T combine(T a,T b) const {return a+b;}
  __device__ T warp_shfl_down(T a,int delta) const {return WARP_SHFL_DOWN(a,delta);}
};
template<typename T> struct Max {
  __device__ T operator()(T a,T b) const {return a<b?b:a;}
  __device__ T combine(T a,T b) const {return a<b?b:a;}
  __device__ T warp_shfl_down(T a,int delta) const {return WARP_SHFL_DOWN(a,delta);}
};
template<template<typename> class Op>
__device__ float block_reduce(float value,float *shared,float initial) {
  float result=at::native::cuda_utils::BlockReduce<float,Op<float>>(value,Op<float>(),initial,shared);
  if(threadIdx.x==0) shared[0]=result;
  __syncthreads();
  return shared[0];
}

template<typename T> __device__ T gather_score(const T* scores,const int* owners,int h,int token,int R,int S,int G) {
  int r=owners[(h/G)*S+token];
  return r>=0?scores[(int64_t)h*R+r]:static_cast<T>(-INFINITY);
}

template<bool EXP,typename T,int ILP,bool VECTOR_TAIL>
__device__ float thread_reduce(const T* scores,const int* owners,int h,int R,int S,int G,float maximum,int N) {
  // Preserve native per-thread token order, including row alignment handling.
  int shift=(static_cast<int64_t>(h)*N)%ILP;
  int size=N,base=0,offset=threadIdx.x;
  float value=EXP?0.f:-FLT_MAX;
  auto consume=[&](int token) {
    float x=static_cast<float>(gather_score(scores,owners,h,token,R,S,G));
    if constexpr (EXP) value=value+std::exp(x-maximum);
    else value=::max(value,x);
  };
  // Native shared softmax visits every aligned vector, including the final
  // partial CTA. Generic softmax instead visits that tail in scalar order.
  if constexpr (VECTOR_TAIL) {
    for(int vector=threadIdx.x;vector*ILP<N;vector+=blockDim.x) {
      #pragma unroll
      for(int j=0;j<ILP;++j) consume(vector*ILP+j);
    }
    return value;
  }
  if(shift>0) {
    base-=shift;size+=shift;
    if(offset>=shift && offset<size) consume(base+offset);
    size-=blockDim.x>size?size:blockDim.x;
    base+=blockDim.x;
  }
  int last=size%(ILP*blockDim.x);
  for(;offset*ILP<size-last;offset+=blockDim.x) {
    #pragma unroll
    for(int j=0;j<ILP;++j) consume(base+offset*ILP+j);
  }
  offset=size-last+threadIdx.x;
  for(;offset<size;offset+=blockDim.x) consume(base+offset);
  return value;
}

template<typename T,int ILP,bool VECTOR_TAIL=false>
__global__ void normalize_stats(const T* scores,const int* owners,float* stats,int R,int S,int G) {
  __shared__ float shared[32];
  int h=blockIdx.x;
  float maximum=block_reduce<Max>(thread_reduce<false,T,ILP,VECTOR_TAIL>(scores,owners,h,R,S,G,0.f,S),shared,-FLT_MAX);
  float sum=block_reduce<Add>(thread_reduce<true,T,ILP,VECTOR_TAIL>(scores,owners,h,R,S,G,maximum,S),shared,0.f);
  if(threadIdx.x==0) {stats[2*h]=maximum;stats[2*h+1]=sum;}
}

// Logical width controls PyTorch's row alignment and reduction order. Owner
// addresses still use fixed capacity; the live width never reaches the host.
template<typename T,int ILP>
__global__ void normalize_stats_live(const T* scores,const int* owners,float* stats,
                                    const int64_t* length,int R,int S,int G,int smem_limit) {
  __shared__ float shared[32];
  int64_t logical_width=*length;
  CUDA_KERNEL_ASSERT(logical_width>9216 && logical_width<=S);
  int h=blockIdx.x,N=static_cast<int>(logical_width);
  bool vector_tail=N<smem_limit && N%ILP==0;
  float local_max=vector_tail
      ? thread_reduce<false,T,ILP,true>(scores,owners,h,R,S,G,0.f,N)
      : thread_reduce<false,T,ILP,false>(scores,owners,h,R,S,G,0.f,N);
  float maximum=block_reduce<Max>(local_max,shared,-FLT_MAX);
  float local_sum=vector_tail
      ? thread_reduce<true,T,ILP,true>(scores,owners,h,R,S,G,maximum,N)
      : thread_reduce<true,T,ILP,false>(scores,owners,h,R,S,G,maximum,N);
  float sum=block_reduce<Add>(local_sum,shared,0.f);
  if(threadIdx.x==0) {stats[2*h]=maximum;stats[2*h+1]=sum;}
}

template<typename T>
__global__ void materialize_probability(const T* scores,const int* owners,const float* stats,T* out,const bool* excluded,int R,int S,int G) {
  int h=blockIdx.y;
  int token=blockIdx.x*blockDim.x+threadIdx.x;
  if(token<S) {
    float score=static_cast<float>(gather_score(scores,owners,h,token,R,S,G));
    // Preserve native std::exp/divide and final model-dtype rounding.
    out[(int64_t)h*S+token]=static_cast<T>(std::exp(score-stats[2*h])/stats[2*h+1]);
  }
}
}

template<typename T>
__global__ void compact_probability(const T* scores,const float* stats,T* out,int R) {
  int h=blockIdx.y,r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<R) {
    float x=static_cast<float>(scores[(int64_t)h*R+r]);
    out[(int64_t)h*R+r]=static_cast<T>(std::exp(x-stats[2*h])/stats[2*h+1]);
  }
}

// Fuse per-head probability rounding and GQA reduction without a [H,R]
// probability intermediate. The full token output preserves native TopK ties.
template<typename T>
__global__ void fused_probability_mean(const T* scores,const int* owners,
                                      const float* stats,T* out,const bool* excluded,int R,int S,int G) {
  int kv=blockIdx.y,token=blockIdx.x*blockDim.x+threadIdx.x;
  if(token>=S) return;
  if(excluded && excluded[token]) {
    out[(int64_t)kv*S+token]=static_cast<T>(-INFINITY);
    return;
  }
  int r=owners[kv*S+token];
  float sum=0.f;
  if(r>=0) {
    for(int g=0;g<G;++g) {
      int h=kv*G+g;
      float x=static_cast<float>(scores[(int64_t)h*R+r]);
      T rounded=static_cast<T>(std::exp(x-stats[2*h])/stats[2*h+1]);
      float probability=static_cast<float>(rounded);
      sum+=std::isnan(probability)?0.f:probability;
    }
  }
  out[(int64_t)kv*S+token]=static_cast<T>(sum/G);
}

at::Tensor virtual_compact_softmax_impl(at::Tensor scores,at::Tensor owners,int64_t S,int64_t G,bool fused_mean,at::Tensor excluded,at::Tensor length) {
  TORCH_CHECK(scores.is_cuda() && owners.is_cuda() && scores.device()==owners.device(),"same CUDA device required");
  TORCH_CHECK(scores.is_contiguous() && owners.is_contiguous(),"contiguous inputs required");
  TORCH_CHECK(scores.dim()==2 && owners.dim()==2 && owners.scalar_type()==at::kInt,"matrix scores/int32 owners required");
  TORCH_CHECK(G>0 && scores.size(0)==owners.size(0)*G && owners.size(1)==S,"GQA shape mismatch");
  TORCH_CHECK(S>2048 && S<INT_MAX && scores.size(1)<INT_MAX,"unsupported size");
  TORCH_CHECK(scores.scalar_type()==at::kHalf || scores.scalar_type()==at::kBFloat16 || scores.scalar_type()==at::kFloat,"fp16/bf16/fp32 required");
  TORCH_CHECK(owners.numel()<INT_MAX,"owner indexing exceeds int32 prototype scope");
  TORCH_CHECK(excluded.is_cuda() && excluded.device()==scores.device() && excluded.scalar_type()==at::kBool && excluded.is_contiguous(),"contiguous same-device bool exclusion mask required");
  TORCH_CHECK(excluded.numel()==0 || (excluded.dim()==1 && excluded.numel()==S),"exclusion mask must be empty or token-domain vector");
  if(length.defined()) {
    TORCH_CHECK(length.is_cuda() && length.device()==scores.device() && length.scalar_type()==at::kLong
                && length.numel()==1 && length.is_contiguous(),"live width must be one same-device int64 value");
  }
  c10::cuda::CUDAGuard device_guard(scores.device());
  auto prop=at::cuda::getCurrentDeviceProperties();
  int bytes=scores.element_size(),ilp=16/bytes;
  TORCH_CHECK((S+1023)/1024>=10,"native reference would use register softmax; fallback required");
  bool native_smem=S<(prop->sharedMemPerBlock-128)/bytes && S%ilp==0;

  auto stats=at::empty({scores.size(0),2},scores.options().dtype(at::kFloat));
  auto out=fused_mean ? at::empty({owners.size(0),S},scores.options())
                      : at::empty({scores.size(0),scores.size(1)},scores.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,scores.scalar_type(),"virtual_softmax",[&] {
    constexpr int ILP=16/sizeof(scalar_t);
    if(length.defined())
    normalize_stats_live<scalar_t,ILP><<<scores.size(0),1024,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),length.data_ptr<int64_t>(),scores.size(1),S,G,(prop->sharedMemPerBlock-128)/bytes);
    else if(native_smem)
    normalize_stats<scalar_t,ILP,true><<<scores.size(0),1024,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),scores.size(1),S,G);
    else
    normalize_stats<scalar_t,ILP><<<scores.size(0),1024,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),scores.size(1),S,G);
    if(fused_mean) fused_probability_mean<scalar_t><<<dim3((S+255)/256,owners.size(0)),256,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),out.data_ptr<scalar_t>(),excluded.numel()?excluded.data_ptr<bool>():nullptr,scores.size(1),S,G);
    else if(scores.size(1)>0) compact_probability<scalar_t><<<dim3((scores.size(1)+255)/256,scores.size(0)),256,0,stream>>>(scores.data_ptr<scalar_t>(),stats.data_ptr<float>(),out.data_ptr<scalar_t>(),scores.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor virtual_compact_softmax(at::Tensor scores,at::Tensor owners,int64_t S,int64_t G,bool fused_mean,at::Tensor excluded) {
  return virtual_compact_softmax_impl(scores,owners,S,G,fused_mean,excluded,at::Tensor());
}
at::Tensor virtual_compact_softmax_live(at::Tensor scores,at::Tensor owners,int64_t S,int64_t G,bool fused_mean,at::Tensor excluded,at::Tensor length) {
  return virtual_compact_softmax_impl(scores,owners,S,G,fused_mean,excluded,length);
}
