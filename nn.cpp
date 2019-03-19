#include  "rand.h"
#include "mat.h"
#include <math.h>

#define weightInitMax 1.000
#define weightInitMin -1.000
#define TRANSFER_SLOPE 2.50
#define ITERATIONS 50000
#define ETA 0.200

void train(int numSteps, int numStrides, int numHidNodes, Matrix& mV, Matrix& mW);
void predict(int numSteps, int numStrides, int numHidNodes, Matrix mV, Matrix mW);

double transfer(double x){
	return 1/(1.0 + exp(-1.0 * TRANSFER_SLOPE * x));
}

double step(double x){
	if(x > 0.5){
		return 1.0;
	}else{
		return 0.0;
	}
}

int main(){

	//Initialize randomization function for weight initialization
	initRand();

	//read in number of steps
	int numSteps, readStatus;
	readStatus = scanf("%d", &numSteps);

	if( readStatus == 0 ){
		printf("Error Reading Input\n");
		exit(-1);
	}

	//Read in number of strides
	int numStrides;
	readStatus = scanf("%d", &numStrides);

	if( readStatus == 0 ){
		printf("Error Reading Input\n");
		exit(-1);
	}

	//Read in number of hidden nodes
	int numHidNodes;
	readStatus = scanf("%d", &numHidNodes);

	if( readStatus == 0 ){
		printf("Error Reading Input\n");
		exit(-1);
	}
	
	//Train on data from stdin and return weights
	Matrix mW, mV;

	train(numSteps, numStrides, numHidNodes, mV, mW);
	
	//Test with trained weights on new data from stdin
	printf("BEGIN TESTING\n");
	predict(numSteps, numStrides, numHidNodes, mV, mW);

	return 0;
}

void predict(int numSteps, int numStrides, int numHidNodes, Matrix mV, Matrix mW){

	//Read in training matrix data
	Matrix mIn("Raw Training Data");
	mIn.read();

	Matrix mX = mIn.seriesSampleCol(0, numSteps, numStrides);
	mX.setName("mX");
	mX.print();
	return;

	Matrix mT(2,2,2);

	//Normalize mX
	mX.normalizeCols();

	//Add Bias col to mX (as mXb)
	Matrix mXb(mX.numRows(), mX.numCols() + 1, 1.0);
	mXb.setName("mX with Bias");
	mXb.insert(mX, 0, 0);

	//Create hidden layer and hidden layer with bias
	Matrix mH(mX.numRows(), numHidNodes, 0.0);
	mH.setName("mH");

	Matrix mHb(mH.numRows(), mH.numCols() + 1, 1.0);
	mHb.setName("mH with Bias");
	mHb.insert(mH, 0, 0);

	//Compute output for prediction
	//compute H
	mH = mXb.dot(mV);
	mH.setName("Hidden Layer");

	//Apply transfer function to H
	mH.map(transfer);

	//Update mHb
	mHb.insert(mH, 0, 0);

	//compute Y
	Matrix mY = mHb.dot(mW);
	mY.setName("Output");

	//Apply transfer function to Y
	mY.map(transfer);

	//Print output to assignment specs
	mY.map(step);
	printf("Target\n");
	for(int i = 0; i < mT.numRows(); i++){
		mT.writeLine(i);
		printf("\n");
	}
	printf("Predicted\n");
	for(int i = 0; i < mY.numRows(); i++){
		mY.writeLine(i);
		printf("\n");
	}

	return;
}

void train(int numSteps, int numStrides, int numHidNodes, Matrix& mV, Matrix& mW){

	//Read in training matrix data
	Matrix mIn("Raw Training Data");
	mIn.read();

	Matrix mX = mIn.seriesSampleCol(0, numSteps, numStrides);
	mX.setName("mX");
	mX.print();
	return;

	Matrix mT(2,2,2);

	//Normalize mX
	mX.normalizeCols();

	//Add Bias col to mX (as mXb)
	Matrix mXb(mX.numRows(), mX.numCols() + 1, 1.0);
	mXb.setName("mX with Bias");
	mXb.insert(mX, 0, 0);

	//Create hidden layer and hidden layer with bias
	Matrix mH(mX.numRows(), numHidNodes, 0.0);
	mH.setName("mH");

	Matrix mHb(mH.numRows(), mH.numCols() + 1, 1.0);
	mHb.setName("mH with Bias");
	mHb.insert(mH, 0, 0);

	//Create Weights for X->H
	mV = Matrix(mXb.numCols(), mH.numCols(), 2.0); //2.0 is arbitrary and is overwritten
	mV.setName("Weights V (X->H)");
	mV.rand(weightInitMin, weightInitMax);

	//Create Weights for H->Y
	mW = Matrix(mHb.numCols(), mT.numCols(), 2.0); //2.0 is arbitrary and is overwritten
	mW.setName("Weights W (H->Y)");
	mW.rand(weightInitMin, weightInitMax);

	for(int i = 0; i < ITERATIONS; i++){
		//compute H
		mH = mXb.dot(mV);
		mH.setName("Hidden Layer");

		//Apply transfer function to H
		mH.map(transfer);

		//Update mHb
		mHb.insert(mH, 0, 0);

		//compute Y
		Matrix mY = mHb.dot(mW);
		mY.setName("Output");
	
		//Apply transfer function to Y
		mY.map(transfer);

		//Calculate dy cost
		Matrix mdY(mY);
		Matrix mYmTdiff(mY);
		mYmTdiff.sub(mT);
		mdY.mul(mYmTdiff);
		Matrix mYtemp(mY);
		mYtemp.scalarPreSub(1.0);
		mdY.mul(mYtemp);

		//Calculate dhb cost
		Matrix mdHb(mHb);
		Matrix mHbtemp(mHb);
		mHbtemp.scalarPreSub(1.0);
		mdHb.mul(mHbtemp);
		mdHb.mul(mdY.dotT(mW));
	
		Matrix mdH(mH);
		mdH.insert(mdHb, 0, 0);

		//Update mW
		mW.sub( (mHb.Tdot(mdY)).scalarMul(ETA) );

		//Update mW
		mV.sub( (mXb.Tdot(mdH)).scalarMul(ETA) );
	
	}
	
	return;
}
