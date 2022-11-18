#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>

#include <OpenCL/opencl.h>


#define PRINT(val,type) printf("%s:\t" type "\n",#val,val);

#define STRINGIFY(code...) #code


float itime(){
    struct timeb now;
    ftime(&now);
    return (float)(now.time%(60*60*24))+now.millitm/1e3;
}

void bubble_sort(int *array,int len){
    for(int j=0;j<len;j++){
        for(int i=0;i<len-1;i++){
            if(array[i]>array[i+1]){
                int tmp=array[i];
                array[i]=array[i+1];
                array[i+1]=tmp;
            }
        }
    }
}

void gpu_bubble_sort(int *array,int len){
    if(len<=1) return;
    int32_t err;
    cl_device_id dev;
    err=clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    assert(err==0);
    cl_context context=clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    assert(err==0);
    cl_command_queue comqueue=clCreateCommandQueue(context,dev,0,&err);
    assert(err==0);
    const char src_arr[]=STRINGIFY(
         __kernel void sort(__global int *array,int off){
             int i=2*get_global_id(0)+off;
             if(array[i]>array[i+1]){
                 int tmp=array[i];
                 array[i]=array[i+1];
                 array[i+1]=tmp;
             }
         }
    );
    unsigned long n=sizeof(src_arr);
    const char *src=&src_arr[0];
    cl_program program=clCreateProgramWithSource(context,1,&src,&n,&err);
    assert(err==0);
    err=clBuildProgram(program,1,&dev,NULL,NULL,NULL);
    if(err!=0){
        size_t len;
        char buf[2048];
        printf("error\n");
        clGetProgramBuildInfo(program,dev,CL_PROGRAM_BUILD_LOG,sizeof(buf),buf,&len);
        printf("%s\n", buf);
        exit(1);
    }
    cl_kernel kernel=clCreateKernel(program,"sort", &err);
    assert(err==0);
    cl_mem mem=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int)*(len),NULL,&err);
    assert(err==0);
    err=clEnqueueWriteBuffer(comqueue,mem,CL_TRUE,0,sizeof(int)*len,array,0,NULL,NULL);
    assert(err==0);
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&mem);
    assert(err==0);
    size_t local=1;
    for(int i=0;i<len-(len==2);i++){
        int off=i%2;
        err=clSetKernelArg(kernel,1,sizeof(int),&off);
        assert(err==0);
        
        size_t global=len/2-(off&&(len%2==0));
        err=clEnqueueNDRangeKernel(comqueue,kernel,1,NULL,&global,&local,0,NULL,NULL);
        assert(err==0);
    }
    clFinish(comqueue);
    err=clEnqueueReadBuffer(comqueue,mem,CL_TRUE,0,sizeof(int)*len,array,0,NULL,NULL);
    assert(err==0);
    clReleaseMemObject(mem);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(comqueue);
    clReleaseContext(context);
}

void merge_sort_rec(int *a,int *c,int n){
    if(n<=1) return;
    merge_sort_rec(a,c,n/2);
    merge_sort_rec(a+n/2,c+n/2,n-n/2);
    int j=0,k=n/2;
    for(int i=0;i<n;i++){
        if(j<n/2 && (a[j]<a[k] || k==n)){
            c[i]=a[j];
            j++;
        }else{
            c[i]=a[k];
            k++;
        }
    }
    for(int i=0;i<n;i++) a[i]=c[i];
}

void merge_sort(int *a,int n){
    int *c=(int*)malloc(n*sizeof(int));
    merge_sort_rec(a,c,n);
    free(c);
}

int ilog2(int x){
    int log=-1;
    while(x){
        x>>=1;
        log++;
    }
    return log;
}

int ipow2(int x){
    return 1<<x;
}

__attribute__((constructor)) void test_i2(){
    puts(__func__);
    assert(ilog2(1)==0);
    assert(ilog2(2)==1);
    assert(ilog2(4)==2);
    assert(ilog2(8)==3);
    
    assert(ilog2(3)==1);
    assert(ilog2(5)==2);
    assert(ilog2(6)==2);
    assert(ilog2(7)==2);
    
    assert(ipow2(0)==1);
    assert(ipow2(1)==2);
    assert(ipow2(2)==4);
    assert(ipow2(3)==8);
    assert(ipow2(4)==16);
    
    assert(ipow2(ilog2(128))==128);
}

void gpu_merge_sort(int *array,int len){
    if(len<=1) return;
    int32_t err;
    cl_device_id dev;
    err=clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    assert(err==0);
    cl_context context=clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    assert(err==0);
    cl_command_queue comqueue=clCreateCommandQueue(context,dev,0,&err);
    assert(err==0);
    const char src_arr[]=STRINGIFY(
        __kernel void sort(__global int *array,__global int *scratch,const int n){
            const int off=get_global_id(0)*n;
            __global int *a=array+off;
            __global int *c=scratch+off;
            const int n2=n/2;
            const int top_half=get_global_id(1);
            const int bot_half=!top_half;
            
            int j=top_half*(n2-1);
            int k=bot_half?n2   :n-1;
            const int istart    =bot_half?0  :n-1;
            const int iend      =bot_half?n2 :n2-1;
            const int istep     =bot_half?1  :-1;

            for(int i=istart;i!=iend;i+=istep){
                const bool jcmpk    =bot_half?a[j]<a[k]:a[j]>a[k];
                const bool kdone    =bot_half?k==n:k==n2-1;
                const bool jndone   =bot_half?j<n2:j>=0;
                if(jndone && (jcmpk || kdone)){
                    c[i]=a[j];
                    j+=istep;
                }else{
                    c[i]=a[k];
                    k+=istep;
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            const int iend2=top_half?n:n2;
            for(int i=n2*top_half;i<iend2;i++) a[i]=c[i];
         }
    );
    unsigned long array_size=sizeof(src_arr);
    const char *src=&src_arr[0];
    cl_program program=clCreateProgramWithSource(context,1,&src,&array_size,&err);
    assert(err==0);
    err=clBuildProgram(program,1,&dev,NULL,NULL,NULL);
    if(err!=0){
        size_t len;
        char buf[2048];
        printf("error\n");
        clGetProgramBuildInfo(program,dev,CL_PROGRAM_BUILD_LOG,sizeof(buf),buf,&len);
        printf("%s\n", buf);
        exit(1);
    }
    cl_kernel kernel=clCreateKernel(program,"sort", &err);
    assert(err==0);
    
    int len_pow2=len;
    
    if(ipow2(ilog2(len))!=len){
        len_pow2=2*ipow2(ilog2(len));
    }
    
    cl_mem array_mem=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int)*(len_pow2),NULL,&err);
    assert(err==0);
    cl_mem scratch_mem=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int)*(len_pow2),NULL,&err);
    assert(err==0);
    
    err=clEnqueueWriteBuffer(comqueue,array_mem,CL_TRUE,0,sizeof(int)*len,array,0,NULL,NULL);
    assert(err==0);
    
    cl_event finish_padding;
    uint32_t num_events=0;
    if(len!=len_pow2){
        int int_max=__INT_MAX__;
        err=clEnqueueFillBuffer(comqueue,array_mem,&int_max,sizeof(int),sizeof(int)*len,
                                sizeof(int)*(len_pow2-len),0,NULL,&finish_padding);
        assert(err==0);
        num_events=1;
    }
      
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&array_mem);
    assert(err==0);
    err=clSetKernelArg(kernel,1,sizeof(cl_mem),&scratch_mem);
    assert(err==0);
    
    size_t max_local;
    err=clGetKernelWorkGroupInfo(kernel,dev,CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(size_t),&max_local,NULL);
    
    assert(err==0);
    PRINT(max_local,"%zu");
    max_local/=2;

    for(int n=2;n<=len_pow2;n*=2){
        err=clSetKernelArg(kernel,2,sizeof(int),&n);
        assert(err==0);
        
        size_t size=(len_pow2)/n;
        size_t global[]={size,2};
        size_t local[]={
            1,
//            size<max_local?size:max_local,
            2,
        };
        err=clEnqueueNDRangeKernel(comqueue,kernel,2,NULL,global,local,
                                   num_events,num_events?&finish_padding:NULL,NULL);
        assert(err==0);
        num_events=0;
    }
    
    float t0=itime();
    clFinish(comqueue);
    float t1=itime();
    printf("clFinish took %f s\n",t1-t0);
    
    err=clEnqueueReadBuffer(comqueue,array_mem,CL_TRUE,0,sizeof(int)*len,array,0,NULL,NULL);
    assert(err==0);
    
    
    clReleaseMemObject(array_mem);
    clReleaseMemObject(scratch_mem);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(comqueue);
    clReleaseContext(context);
}


int main(){
    puts(__func__);

    const int len=10000;
    {
        int *array=(int*)malloc(len*sizeof(int));

        for(int i=0;i<len;i++) array[i]=len-i-1;

        float t0=itime();
        gpu_merge_sort(array,len);
        float t1=itime();
        printf("gpu_merge_sort took %f s\n",t1-t0);
//        for(int i=0;i<len;i++){
//            printf("%d,\t",array[i]);
//        }
//        puts("");
        for(int i=0;i<len-1;i++){
            assert(array[i]==i);
        }
        free(array);
    }
    {
        int *array=(int*)malloc(len*sizeof(int));
        
        for(int i=0;i<len;i++) array[i]=len-i-1;
        
        float t0=itime();
        merge_sort(array,len);
        float t1=itime();
        printf("merge_sort took %f s\n",t1-t0);
        for(int i=0;i<len-1;i++){
            assert(array[i]==i);
        }
        free(array);
    }
    {
        int *array=(int*)malloc(len*sizeof(int));

        for(int i=0;i<len;i++) array[i]=len-i-1;

        float t0=itime();
        gpu_bubble_sort(array,len);
        float t1=itime();
        printf("gpu_bubble_sort took %f s\n",t1-t0);


        for(int i=0;i<len-1;i++){
            assert(array[i]<array[i+1]);
        }

        free(array);
    }
    {
        int *array=(int*)malloc(len*sizeof(int));

        for(int i=0;i<len;i++) array[i]=len-i-1;

        float t0=itime();
        bubble_sort(array,len);
        float t1=itime();
        printf("bubble_sort took %f s\n",t1-t0);
        for(int i=0;i<len-1;i++){
            assert(array[i]<array[i+1]);
        }
        free(array);
    }
}
