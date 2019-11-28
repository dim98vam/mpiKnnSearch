#include <stdio.h>
#include "cblas.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "knnring.h"
#include "mpi.h"
#include <float.h>



//SWAP for elements in partition
#define SWAP(x, y) { double temp = *x; *x = *y; *y = temp; }

#define SWAPDouble(x, y) { double temp = x; x = y; y = temp; }
#define SWAPint(x,y){int temp = x; x = y; y = temp;}

// Partition using Lomuto partition scheme and parallel update of the initial index table
//validated
double* partition(int size,int* a,double* left, double* right, double* pivotIndex)
{
	// Pick pivotIndex as pivot from the array
	double pivot = *pivotIndex;
	int* pivotA = a+(pivotIndex-left);

	// Move pivot to end
	SWAP(pivotIndex, right);
	SWAP(pivotA,(a+size-1));

	// elements less than pivot will be pushed to the left of pIndex
	// elements more than pivot will be pushed to the right of pIndex
	// equal elements can go either way
	pivotIndex = left;
	pivotA=a;

	int i;

	// each time we finds an element less than or equal to pivot, pIndex
	// is incremented and that element would be placed before the pivot and index table is also updated.
	for (i = 0; i < size-1; i++)
	{
		if (left[i] <= pivot)
		{
			SWAP((left+i), pivotIndex);
			SWAP((a+i),pivotA);
			pivotIndex++;
			pivotA++;
		}
	}

	// Move pivot to its final place
	SWAP(pivotIndex, right);
	SWAP(pivotA,(a+size-1));

	// return pIndex (index of pivot element)
	return pivotIndex;
}

// Returns the k-th smallest element of list within left..right
// (i.e. left <= k <= right) while updating initial index table using lomuto partition
//validated
double* quickselect(int size,int* a,double* left, double* right, int k)
{
	// If the array contains only one element, return that element
	if (left == right)
		return left;

	// select a pivotIndex between left and right
	double*  pivotIndex = left + (rand() % (right - left + 1));

	pivotIndex = partition(right-left+1,a, left, right, pivotIndex);

	// The pivot is in its final sorted position
	if ((left+k-1) == pivotIndex)
		return pivotIndex;

	// if k is less than the pivot index
	else if ( (left+k-1)< pivotIndex)
		return quickselect((pivotIndex-left), a, left, pivotIndex - 1, k);

	// if k is more than the pivot index
	else
		return quickselect(right-pivotIndex,a+(pivotIndex-left)+1, pivotIndex + 1, right, k-(pivotIndex-left)-1);
}

//quicksort used to reorder k smallest distances in ascending order while updating the index table
void quickSort(int* a, double* low, double* high)
{
    if (low < high)
    {
        //create the partition for input tables
        double*  pivotIndex = low + (rand() % (high - low));
        pivotIndex = partition(high-low+1,a, low, high, pivotIndex);

        // Separately sort elements before
        // partition and after partition
        quickSort(a, low, pivotIndex - 1);
        quickSort(a+(pivotIndex-low)+1, pivotIndex + 1, high);
    }
}




//function to calculate matrix of distances. Size of result is mxn
void calculateDistances(double *X, double *Y, int n, int m, int d,double** pointer)
{
	// Calculate X.X**T
	double *dot_X = (double*)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		*(dot_X + i) = cblas_ddot(d, (X + i * d), 1, (X + i * d), 1);
	// Calculate Y.Y**T
	double *dot_Y = (double*)malloc(m * sizeof(double));
	for (int i = 0; i < m; i++)
		*(dot_Y + i) = cblas_ddot(d, (Y + i * d), 1, (Y + i * d), 1);
	// Calculate (X.X**T + Y.Y**T)
	double *D = (double*)calloc(m*n, sizeof(double));
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			*(D + i * n + j) = *(dot_Y + i) + *(dot_X + j);
	// Calculate 2.Y.X**T and all distances
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2.0, Y, d, X, d, 1.0, D, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			if (*(D + i * n + j) < 1e-12)	// threshold min value for openblas
				*(D + i * n + j) = 0.0;
			else
				*(D + i * n + j) = sqrt(*(D + i * n + j));
		}
	*(pointer)=D;
}

knnresult kNN(double* X,double* Y,int n,int m,int d,int k){


        //calculate distanceMatrix
        double* distanceMatrix;
        calculateDistances(X,Y,n,m,d,&distanceMatrix);

        //find and order in ascending manner the k smallest elements for its row
        //of the distanceMatrix

        /*create index table for all rows*/
        int* index=(int*)malloc(m*n*sizeof(int));
        int i;
        int t;
        for(i=0;i<m;++i)
            for(t=0;t<n;t++)
                index[i*n+t]=t;

        /*order elements and return resulting tables in knnstruct*/
        double* kDistancesTable=(double*)malloc(m*k*sizeof(double));
        int* kIndex=(int*)malloc(m*k*sizeof(int));
        double* kthElement;
        int y;
        for(i=0;i<m;++i){
            //order only the necessary k elements to save time
            kthElement=quickselect(n,index+i*n,distanceMatrix+i*n,distanceMatrix+i*n+(n-1),k);
            quickSort(index+i*n,distanceMatrix+i*n,kthElement);

            //copy k elements of row i into result table row i
            for(y=0;y<k;++y){
                kDistancesTable[i*k+y]= distanceMatrix[i*n+y];
                kIndex[i*k+y]=index[i*n+y];
            }
        }

        knnresult* result=(knnresult*)malloc(sizeof(knnresult));
        result->nidx=kIndex;
        result->ndist=kDistancesTable;
        result->k=k;
        result->m=m;

        return *result;
}

// Merge ar1[] and ar2[]
void merge(double* ar1, double* ar2,int* ind1,int* ind2, int k,int pid) {

    //check for the first k smallest elements
    int* tempInd = (int*)malloc(k*sizeof(int));
    double* tempDist = (double*)malloc(k*sizeof(double));
    int i = 0;
    int j = 0;
    int t = 0;
    while (t<k)
    {
        if (ar1[i] <= ar2[j])
        {
            tempDist[t] = ar1[i];
            tempInd[t]=ind1[i];
            ++i;
        }
        else
        {
            tempDist[t] = ar2[j];
            tempInd[t]=ind2[j];
            ++j;
        }
        ++t;
    }

   //copy them into answer table
   memcpy(ar1,tempDist,k*sizeof(double));
   memcpy(ind1,tempInd,k*sizeof(int));
}

//function to calculate new knnresult struct after given received index and points tables
void kNNMpi(double* X,double* Y,double* kNeighborsDist,int* indx ,int n,int m,int d,int k,knnresult* holder,int pid){

        //calculate distanceMatrix
        double* distanceMatrix;
        calculateDistances(X,Y,n,m,d,&distanceMatrix);
        //find and order in ascending manner the k smallest elements for its row
        //of the distanceMatrix
        int* index=(int*)malloc(m*n*sizeof(int));
        int i;
        for(i=0;i<m;++i)
            memcpy(&index[n*i],indx,n*sizeof(int));

        /*order elements and return resulting tables in knnstruct*/
        double* kthElement;
        for(i=0;i<m;++i){
            //order only the necessary k elements to save time
            kthElement=quickselect(n,index+i*n,distanceMatrix+i*n,distanceMatrix+i*n+(n-1),k);
            quickSort(index+i*n,distanceMatrix+i*n,kthElement);

            //merge k elements of row i into result table row i of holder
            merge((holder->ndist)+i*k,distanceMatrix+i*n,(holder->nidx)+i*k,index+i*n,k,pid);
        }

}

//function to calculate knnresult an pass points around in ring fashion using mpi protocol
//for communication in synchronous way
knnresult distrAllkNN(double* X,int n,int d,int k){
   int pid;
   int nproc;
   MPI_Status status[4];
   int tag1=3;
   int tag2=2;
   MPI_Request reqs[4];
   knnresult* finalResult;
   double* Y;

   //define process' pid and number of processes in total
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);

   //define tables to hold both incoming and outgoing messages in order to enable faster
   //asynchronous com and simpler code in the expense of space
   Y=(double*)malloc(n*d*sizeof(double));
   double* XtoCome=(double*)malloc(n*d*sizeof(double));
   int* indexes=(int*)malloc(n*sizeof(int));
   int* indexesToCome=(int*)malloc(n*sizeof(int));
   memcpy(Y,X,n*d*sizeof(double));

   //calculate indexes in accordance to the table created in tester's main
   int i;
   if(pid!=0){
      for(i=0;i<n;++i)
         indexes[i]=(pid-1)*n+i;
   }
   else{
      for(i=0;i<n;++i)
          indexes[i]=(nproc-1)*n+i;
   }

    /*initialize final result*/
    finalResult=(knnresult*)malloc(sizeof(knnresult));

    //initialize all distances to infinity at first
    finalResult->ndist=(double*)malloc(n*k*sizeof(double));
    for(i=0;i<n*k;++i)
         finalResult->ndist[i]=DBL_MAX;


    //just allocate for index table
    finalResult->nidx=(int*)malloc(n*k*sizeof(int));
    for(i=0;i<n*k;++i)
         finalResult->nidx[i]=-1;

    finalResult->m=n;
    finalResult->k=k;

    //start passing around points and indexes in ring manner in asynchronous way
    //before actual calculations to allow fro pipelining of the toy above
    for(i=0;i<nproc-1;++i){

       //send info forward
       MPI_Isend(X,n*d,MPI_DOUBLE,(pid==nproc-1)?0:(pid+1),tag1,MPI_COMM_WORLD,&reqs[0]);
       MPI_Isend(indexes,n,MPI_INT,(pid==nproc-1)?0:(pid+1),tag2,MPI_COMM_WORLD,&reqs[1]);

       //receive points for next calculation
       MPI_Irecv(XtoCome, n*d, MPI_DOUBLE,(pid==0)?nproc-1:(pid-1), tag1, MPI_COMM_WORLD,&reqs[2]);
       MPI_Irecv(indexesToCome, n, MPI_INT, (pid==0)?nproc-1:(pid-1), tag2, MPI_COMM_WORLD,&reqs[3]);

       //calculate new distances and update finalResult
       kNNMpi(X,Y,finalResult->ndist,indexes ,n,n,d,k,finalResult,pid);


       MPI_Waitall(4, reqs, status);





       //copy X and indexes received into outgoing buffers
       memcpy(X,XtoCome,n*d*sizeof(double));
       memcpy(indexes,indexesToCome,n*sizeof(double));
       }

       //calculate new distances and update finalResult for last elements received
       kNNMpi(X,Y,finalResult->ndist,indexes ,n,n,d,k,finalResult,pid);

       return *finalResult;
}
