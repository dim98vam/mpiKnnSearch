#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utility.h"
#include "mpi.h"

void parallelKnn(int argc, char** argv,int n,int d,int k){
   int pid;
   int nproc;
   MPI_Status status[4];
   int tag1=1;
   int tag2=2;
   MPI_Request reqs[4];
   int count;
   knnresult finalResult;
   double* Y;

   //starting mpi communication
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);

   //define tables to hold both incoming and outgoing messages in order to enable faster
   //asynchronous com and simpler code in the expense of space
   double* X=(double*)malloc((n/nproc+n%nproc)*d*sizeof(double));
   double* XtoCome=(double*)malloc((n/nproc+n%nproc)*d*sizeof(double));
   int* indexes=(int*)malloc((n/nproc+n%nproc)*sizeof(int));
   int* indexesToCome=(int*)malloc((n/nproc+n%nproc)*sizeof(int));

   if(pid!=0){

    //define table to hold query points of process
    Y=(double*)malloc((n/nproc)*d*sizeof(double));

    //receive Y and indexes initially and calculate initial knnresult
    MPI_Irecv(Y, n/nproc*d, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD,&reqs[0]);
    MPI_Irecv(indexes, n/nproc, MPI_INT, 0, tag2, MPI_COMM_WORLD,&reqs[1]);

    MPI_Waitall(2, reqs, status);

    finalResult= kNNSequential(Y,Y,indexes,n/nproc,n/nproc,d,k);
    memcpy(X,Y,(n/nproc)*d*sizeof(double));

    //start passing around points and indexes in ring manner in asynchronous way

    count=n/nproc;
   }
   //perform root initial operations
   else{

     //initialize X and index table
     Y=(double*)malloc(n*d*sizeof(double));
     int* initialIndex=(int*)malloc(n*sizeof(int));

     int y;
     for(y=0;y<n*d;++y)
        Y[y]=(double)rand()/(double)RAND_MAX;
     for(y=0;y<n;++y)
        initialIndex[y]=y;

     //scatter the above among the processes
     int initialStep=n%nproc+n/nproc;
     int count1=n/nproc;

     for(y=0;y<nproc-1;y++){
         MPI_Isend(Y+initialStep*d+y*count1*d,count1*d,MPI_DOUBLE,y+1,tag1,MPI_COMM_WORLD,&reqs[0]);
         MPI_Isend(initialIndex+initialStep+y*count1,count1,MPI_INT,y+1,tag2,MPI_COMM_WORLD,&reqs[1]);

         MPI_Waitall(2, reqs, status);

     }
     //calculate initial knnresult for root
     finalResult= kNNSequential(Y,Y,initialIndex,initialStep,initialStep,d,k);
     memcpy(X,Y,initialStep*d*sizeof(double));
     memcpy(indexes,initialIndex,initialStep*sizeof(int));
     free(initialIndex);
     Y=realloc(Y,initialStep*d*sizeof(double));

     count=initialStep;
   }

    //start passing parts of initialX and indexes around in the ring
    int j;
    for(j=0;j<nproc-1;++j){
       //send info forward
       MPI_Isend(X,count*d,MPI_DOUBLE,(pid==nproc-1)?0:(pid+1),tag1,MPI_COMM_WORLD,&reqs[0]);
       MPI_Isend(indexes,count,MPI_INT,(pid==nproc-1)?0:(pid+1),tag2,MPI_COMM_WORLD,&reqs[1]);

       //check for number of elements coming
       MPI_Probe(pid-1,tag2,MPI_COMM_WORLD,&status[1]);

       //check whether more than n/nproc elements are to be received
       MPI_Get_count(&status[1], MPI_INT, &count);

       MPI_Irecv(XtoCome, count*d, MPI_DOUBLE, pid-1, tag1, MPI_COMM_WORLD,&reqs[2]);
       MPI_Irecv(indexesToCome, count, MPI_INT, pid-1, tag2, MPI_COMM_WORLD,&reqs[3]);

       MPI_Waitall(4, reqs, status);
       //calculate new distances and update finalResult
       kNNMpi(XtoCome,Y,finalResult.ndist,indexesToCome ,count,(pid==0)?n%nproc+n/nproc:n/nproc,d,k,&finalResult);

       //copy X and indexes into outgoing buffers
       memcpy(X,XtoCome,count*d*sizeof(double));
       memcpy(indexes,indexesToCome,count*sizeof(double));
    }

   MPI_Finalize();
}

//function for testing
int main(int argc, char** argv){
    parallelKnn(argc,argv,5,3,2);
    return 0;

}
