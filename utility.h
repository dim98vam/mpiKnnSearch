#ifndef UTILITY_H
#define UTILITY_H

typedef struct knnresult{
  int* nidx;
  double* ndist;
  int m;
  int k;
}knnresult;

// Partition using Lomuto partition scheme and parallel update of the initial index table
//validated
double* partition(int size,int* a,double* left, double* right, double* pivotIndex);

// Returns the k-th smallest element of list within left..right
// (i.e. left <= k <= right) while updating initial index table using lomuto partition
//validated
double* quickselect(int size,int* a,double* left, double* right, int k);


// Returns the k-th smallest element of list within left..right
// (i.e. left <= k <= right) while updating initial index table using lomuto partition
//validated
double* quickselect(int size,int* a,double* left, double* right, int k);

//quicksort used to reorder k smallest distances in ascending order while updating the index table
void quickSort(int* a, double* low, double* high);

//function used to obtain a 1xn matrix from nxd, calculating the dot product squared of its rows and returning a row vector
double* DotRowOfMatrix(int rows,int columns,double *matrix);

//function to add two vectors, first one of size 1xM and the second one of 1xn where M<n creating an
//Mxn matrix ---- blocking technic applied to obtain better performance
double* vectorMatrixAddition(double* A,double* B,int m,int n);

//function to calculate matrix of distances. Size of result is mxn
void calculateDistances(double* X,double* Y,int n,int m,int d,double** pointer);

//function to calculate new knnresult struct initially
knnresult kNNSequential(double* X,double* Y,int* indexes,int n,int m,int d,int k);

//functions for merging new with old index and distance tables without the use of extra space
int nextGap(int gap);
void merge(double* arr1, double* arr2,int* kIndexes,int* indexes, int n, int m);

//function to calculate new knnresult struct after given received index and points tables
void kNNMpi(double* X,double* Y,double* kNeighborsDist,int* indx ,int n,int m,int d,int k,knnresult* holder);

#endif // UTILITY_H_INCLUDED

/*
//function for testing
int main(void){

    double holder[4][3]={{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    //double b[4][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    double y[2][3]={{13,14,15},{16,17,18}};

    knnresult result=kNN((double*)holder,(double*)y,4,2,3,3);

    int i;
    int j;
    for(i=0;i<2;++i){
       for(j=0;j<3;++j)
         printf("%lf ",(result.ndist)[i*3+j]);

        printf("\n");
    }
    printf("\n");

    for(i=0;i<2;++i){
       for(j=0;j<3;++j)
         printf("%d ",(result.nidx)[i*3+j]);

        printf("\n");
    }
    printf("\n");
    return 0;
}
*/
