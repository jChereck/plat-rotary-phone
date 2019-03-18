#include  "rand.h"
#include "mat.h"
#include <math.h>

#define weightInitMax 1.500
#define weightInitMin -1.500
#define TRANSFER_SLOPE 1.50
#define ITERATIONS 105000
#define ETA 0.300

Matrix train(int numIn, int numHidNodes, Matrix& mV, Matrix& mW);
void predict(int numIn, int numHidNodes, Matrix mV, Matrix mW, Matrix trainMinMax);

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

	//read in number of inputs from stdin
	int numIn, readStatus;
	readStatus = scanf("%d", &numIn);

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

	//Read in number of classes
	int numClasses;
	readStatus = scanf("%d", &numClasses);

	if( readStatus == 0 ){
		printf("Error Reading Input\n");
		exit(-1);
	}
	
	//Train on data from stdin and return weights
	Matrix mW, mV;

	Matrix trainMinMax = train(numIn, numHidNodes, mV, mW);
	
	//Test with trained weights on new data from stdin
	printf("BEGIN TESTING\n");
	predict(numIn, numHidNodes, mV, mW, trainMinMax);

	return 0;
}

void predict(int numIn, int numHidNodes, Matrix mV, Matrix mW, Matrix trainMinMax){

	//Read in training matrix data
	Matrix mIn("Raw Training Data");
	mIn.read();

	//Split raw input into X and T matricies
	Matrix mX = mIn.extract(0, 0, 0, numIn);
	mX.setName("mX");
	Matrix mT = mIn.extract(0, numIn, 0, 0);
	mT.setName("mT");

	//Normalize mX
	mX.normalizeCols(trainMinMax);

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
	Matrix mOut = mY.argMaxRow();
	for(int i = 0; i < mY.numRows(); i++){
		mOut.writeLine(i);
		printf("\n");
	}

	//Print confusion matricies
	
	for(int c = 0; c < mOut.numCols(); c++){
		Matrix cm(mY.numCols(),mY.numCols(),0.0);
		for(int r = 0; r < mY.numRows(); r++){

			cm.inc(mT.get(r,c),mOut.get(r,c));
		}
		cm.printfmt("Confusion Matrix");
		
	}
	

	return;
}

Matrix train(int numIn, int numHidNodes, Matrix& mV, Matrix& mW){

	//Read in training matrix data
	Matrix mIn("Raw Training Data");
	mIn.read();

	//Split raw input into X and T matricies
	Matrix mX = mIn.extract(0, 0, 0, numIn);
	mX.setName("mX");
	Matrix mT = mIn.extract(0, numIn, 0, 0);
	mT.setName("mT");

	//Normalize mX
	Matrix trainMinMax = mX.normalizeCols();

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
	
	return trainMinMax;
}
