#include  "rand.h"
#include "mat.h"
#include <math.h>

#define weightInitMax 1.000
#define weightInitMin -1.000
#define TRANSFER_SLOPE 2.50
#define ITERATIONS 5000
#define ETA 0.0200
#define MOMENTUM 0.1

void train(int numSteps, int numStrides, int numHidNodes, Matrix mIn, Matrix& mV, Matrix& mW);
void predict(int numSteps, int numStrides, int numHidNodes, Matrix mIn ,Matrix mV, Matrix mW);

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

	//Read in training matrix data
	Matrix mIn("Raw Training Data");
	mIn.read();

	train(numSteps, numStrides, numHidNodes, mIn, mV, mW);

	
	//Test with trained weights on new data from stdin
	printf("BEGIN TESTING\n");
	predict(numSteps, numStrides, numHidNodes, mIn, mV, mW);

	return 0;
}

void predict(int numSteps, int numStrides, int numHidNodes, Matrix mIn, Matrix mV, Matrix mW){

	Matrix mIn2 = mIn.seriesSampleCol(0, numSteps, numStrides);

	Matrix mX = mIn2.extract(0, 0, 0, mIn2.numCols() - 1);
	mX.setName("mX");
	Matrix mT = mIn2.extract(0, mIn2.numCols() - 1, 0, 0);
	mT.setName("mT");

	//Normalize mX
	Matrix normalization = mX.normalizeCols();

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

	//Remove transfer function to last layer
	//Apply transfer function to Y
	//mY.map(transfer);

	//mY.normalizeCols(normalization);

	//Print output to assignment specs
	/*
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
	*/

	printf("Predicted\n");
	for(int i = 0; i < mY.numRows(); i++){
		mT.writeLine(i);
		printf(" T --> P ");
		mY.writeLine(i);
		printf("\n");
	}

	return;
}

void train(int numSteps, int numStrides, int numHidNodes, Matrix mIn, Matrix& mV, Matrix& mW){

	Matrix mIn2 = mIn.seriesSampleCol(0, numSteps, numStrides);

	Matrix mX = mIn2.extract(0, 0, 0, mIn2.numCols() - 1);
	mX.setName("mX");
	Matrix mT = mIn2.extract(0, mIn2.numCols() - 1, 0, 0);
	mT.setName("mT");


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

	Matrix mWchangePrev;
	Matrix mVchangePrev;
	bool first = true;

	Matrix numRows(mX.numRows(), 1, mX.numRows());

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
	
		//Remove transfer function to last layer
		//Apply transfer function to Y
		//mY.map(transfer);

		//Calculate dy cost
		Matrix mdY(mY);
		Matrix mYmTdiff(mY);
		mYmTdiff.sub(mT);

		//divide by number of samples
		//mYmTdiff.div(numRows);
		mYmTdiff.scalarMul(1.0/mX.numRows());
		//mYmTdiff.print();

		/*
		mdY.mul(mYmTdiff);
		Matrix mYtemp(mY);
		mYtemp.scalarPreSub(1.0);
		mdY.mul(mYtemp);
		*/

		//Calculate dhb cost
		Matrix mdHb(mHb);
		Matrix mHbtemp(mHb);
		mHbtemp.scalarPreSub(1.0);
		mdHb.mul(mHbtemp);
		mdHb.mul(mdY.dotT(mW));
	
		Matrix mdH(mH);
		mdH.insert(mdHb, 0, 0);

		//Update mW
		Matrix mWchange = mHb.Tdot(mYmTdiff).scalarMul(ETA);
		mWchange.setName("mW change");
		//add momentum
		//mWchange.print();
		if( !first){
			//mWchange.scalarMul(1 - MOMENTUM);
			mWchange.add( mWchangePrev.scalarMul(MOMENTUM) );
		}
		mWchangePrev = mWchange;
		//mWchange.print();

		mW.sub( mWchange );

		//Update mW
		//mW.sub( (mHb.Tdot(mdY)).scalarMul(ETA) );

		//Update mV
		Matrix mVchange = mXb.Tdot(mdH).scalarMul(ETA);
		//add momentum
		if( !first){
			//mVchange.scalarMul(1 - MOMENTUM);
			mVchange.add( mVchangePrev.scalarMul(MOMENTUM) );
		}
		mVchangePrev = mVchange;

		mV.sub( mVchange );

		//Update mV
		//mV.sub( (mXb.Tdot(mdH)).scalarMul(ETA) );

		first = false;

	}

	return;
}
