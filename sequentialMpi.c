#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


//value used for blocking
#define BLOCK 500
//SWAP for elements in partition
#define SWAP(x, y) { double temp = *x; *x = *y; *y = temp; }


typedef struct knnresult{
  int* nidx;
  double* ndist;
  int m;
  int k;
}knnresult;


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
        double*  pivotIndex = low + (rand() % (high - low + 1));
        pivotIndex = partition(high-low+1,a, low, high, pivotIndex);

        // Separately sort elements before
        // partition and after partition
        quickSort(a, low, pivotIndex - 1);
        quickSort(a+(pivotIndex-low)+1, pivotIndex + 1, high);
    }
}

//function used to obtain a 1xn matrix from nxd, calculating the dot product squared of its rows and returning a row vector
double* DotRowOfMatrix(int rows,int columns,double *matrix){
    int i;

    //vector to hold the results
    double* result=(double*)malloc(rows*sizeof(double));

    for(i=0;i<rows;++i){
       gsl_vector_view A = gsl_vector_view_array((double*)(matrix+i*columns),columns);
       //get the dot product of each row with itself
       gsl_blas_ddot(&A.vector,&A.vector , result+i);
    }

    return result;
}

//function to add two vectors, first one of size 1xM and the second one of 1xn where M<n creating an
//Mxn matrix ---- blocking technic applied to obtain better performance
double* vectorMatrixAddition(double* A,double* B,int m,int n){

    double* result=(double*)malloc(m*n*sizeof(double));
    int i;
    int jj;
    int j;

    //for loops to initialize result table
    for(jj=0;jj<n;jj=jj+BLOCK){
        for(i=0;i<m;++i){
            for(j=jj;j<((jj+BLOCK<n)?jj+BLOCK:n);++j){
                result[i*n+j]=(A[i]+B[j]);
            }
        }
    }

    return result;
}


//function to calculate matrix of distances. Size of result is mxn
void calculateDistances(double* X,double* Y,int n,int m,int d,double** pointer){
      double* Xdot,*Ydot,*combinedDot;

      //applying the same formula as in the report to obtain the distance matrix using cblas assist
      Xdot=DotRowOfMatrix(n,d,X);
      Ydot=DotRowOfMatrix(m,d,Y);
      combinedDot=vectorMatrixAddition(Ydot,Xdot,m,n);

      gsl_matrix_view A = gsl_matrix_view_array(X, n, d);
      gsl_matrix_view B = gsl_matrix_view_array(Y, m, d);

      double result[m*n];
      memset(result, 0.0, m*n * sizeof(double));
      gsl_matrix_view C = gsl_matrix_view_array((double*)result, m, n);

      //multiplying the matrices and subtracting from combinedDot matrix
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, -2.0, &B.matrix, &A.matrix, 1.0, &C.matrix);
      A=gsl_matrix_view_array(combinedDot, m, n);
      gsl_matrix_add(&C.matrix, &A.matrix);

      int i,y;
      for(i=0;i<m;++i){
        for(y=0;y<n;++y)
            result[i*n+y]=sqrt(result[i*n+y]);
      }

      (*pointer)=result;

}

knnresult* kNN(double* X,double* Y,int n,int m,int d,int k){

        //calculate distanceMatrix
        double* distanceMatrix;
        calculateDistances(X,Y,n,m,d,&distanceMatrix);

        //find and order in ascending manner the k smallest elements for its row
        //of the distanceMatrix

        /*create index table for all rows*/
        int* index=(int*)malloc(m*n*sizeof(int));
        int i;
        for(i=0;i<m*n;++i)
            index[i]=(i%n);

        /*order elements and return resulting tables in knnstruct*/
        double* kDistancesTable=(double*)malloc(m*k*sizeof(double));
        int* kIndex=(int*)malloc(m*k*sizeof(int));
        double* kthElement;
        int y;
        for(i=0;i<m;++i){
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

        return result;
}







//function for testing
int main(void){

    double holder[4][3]={{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    //double b[4][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    double y[2][3]={{13,14,15},{16,17,18}};

    knnresult* result=kNN((double*)holder,(double*)y,4,2,3,3);

    int i;
    int j;
    for(i=0;i<2;++i){
       for(j=0;j<3;++j)
         printf("%lf ",(result->ndist)[i*3+j]);

        printf("\n");
    }

    printf("\n");

    for(i=0;i<2;++i){
       for(j=0;j<3;++j)
         printf("%d ",(result->nidx)[i*3+j]);

        printf("\n");
    }

    printf("\n");
    return 0;

}
