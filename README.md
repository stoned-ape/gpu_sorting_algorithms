# GPU Sorting Algorithms with OpenCL
## Bubble Sort
For an array of length n you must make n passes.  
Each pass consists of looking at all adjacent elements and swapping them if they are out of order.
This is trival to parrallelize.  

For even numbered passes, you compare and swap elements (0,1),(2,3),(4,5) ... 

For odd numbered passes, you compare and swap elements (1,2),(3,4),(5,6) ...
 
Each compare and swap can be done in a separate thread.
So each pass can be done with n/2 parallel threads.
The algorithm is only partially parallel.
Pass 1 cannot begin until pass 0 has finished.

If you had an infinite number of parallel cores, the time complexity is O(n).
## Merge Sort
For simplicity, my implementation only operates on arrays whose length is a power of 2.
If the provided array's length is not a power of 2, it is expanded to the nearest power of 2 
up and padded with INT\_MAX.

Consider an array A with length 2^3 or 8.  
A merge sort for this array requires the following merges 
```c
int a[8];

merge(a+0,a+1,2);merge(a+2,a+3,2);merge(a+4,a+5,2);merge(a+6,a+7,2);
merge(a+0,a+2,4);merge(a+4,a+6,4);
merge(a+0,a+4,8);
```
Notice that merges on the same line are independed and can be done in parallel.
Merges on different lines are dependent and must be done in the order they are written.

The merge itself can also be parallelized somewhat.  A sequential merge normally looks like this:

```c++
//c is just a scratch buffer
//the two arrays we are merging are a[0:n/2-1] and a[n/2:n-1]
//the result is stored in a
void merge(int *a,int *c,int n){
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
```
Here we are starting at the front and moving forward.  
We could just as easily start from back and move backward. 

```c++
void merge(int *a,int *c,int n){
    int j=n/2-1,k=n-1;
    for(int i=n-1;i>=0;i--){
        if(j>=0 && (a[j]>a[k] || k==n/2-1)){
            c[i]=a[j];
            j--;
        }else{
            c[i]=a[k];
            k--;
        }
    }
    for(int i=0;i<n;i++) a[i]=c[i];
}
```
Why not do both and meet in the middle?
```c++
void merge(int *a,int *c,int n){
    { //merge bottom half
        int j=0,k=n/2;
        for(int i=0;i<n/2;i++){
            if(j<n/2 && (a[j]<a[k] || k==n)){
                c[i]=a[j];
                j++;
            }else{
                c[i]=a[k];
                k++;
            }
        }
    }
    {  //merge top half
        int j=n/2-1,k=n-1;
        for(int i=n-1;i>=n/2;i--){
            if(j>=0 && (a[j]>a[k] || k==n/2-1)){
                c[i]=a[j];
                j--;
            }else{
                c[i]=a[k];
                k--;
            }
        }
    }
    for(int i=0;i<n;i++) a[i]=c[i];
}
```
These two merges can be done in parallel.  
The final copying of c back into a can also be broken into two pieces as 
long as both merges have already completed when you start copying.

Here is the final OpenCL kernel for doing this.
If get\_global\_id(1) returns one, the thread with merge from the top down, 
else it will merge from the bottom up.  
The barrier() function ensures that both halves have been merged into c, 
before c is copied back into a.
```c++
__kernel void sort(__global int *array,__global int *scratch,const int n){
    const int off=get_global_id(0)*n;
    __global int *a=array+off;
    __global int *c=scratch+off;
    const int n2=n/2;
    const int top_half=get_global_id(1);
    const int bot_half=!top_half;
    
    int j=bot_half?0    :(n2-1);
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
```
This code is quite terse. 
Conditional branches have been collapsed into conditional moves 
because of measurable performance gains.
