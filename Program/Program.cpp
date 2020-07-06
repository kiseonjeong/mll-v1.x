// Program.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

// The Test Function List
void test_knn(string, string);
void test_naivebayes(string, string);
void test_normalbayes(string, string);
void test_svm(string, string);
void test_adaboost(string, string);
void test_logitmodel(string, string);
void test_mlp(string, string);
void test_ae(string, string);

// The Main Procedure
int _tmain(int argc, _TCHAR* argv[])
{
	// Do test
	test_knn("..\\..\\dataset\\knn\\datingTestSet.txt", "..\\..\\dataset\\knn\\datingTestSet2.txt");
	test_naivebayes("..\\..\\dataset\\bys\\iris_train.txt", "..\\..\\dataset\\bys\\iris_test.txt");
	test_normalbayes("..\\..\\dataset\\bys\\iris_train.txt", "..\\..\\dataset\\bys\\iris_test.txt");
// 	test_svm("..\\Dataset\\svm\\testSetRBF.txt", "..\\Dataset\\svm\\testSetRBF2.txt");
//	test_adaboost("..\\Dataset\\adaboost\\horseColicTraining2.txt", "..\\Dataset\\adaboost\\horseColicTest2.txt");
// 	test_logitmodel("..\\Dataset\\lgr\\testSet.txt", "..\\Dataset\\lgr\\testSet.txt");
//	test_mlp("..\\Dataset\\mlp\\iris_train.txt", "..\\Dataset\\mlp\\iris_test.txt");
//	test_ae("..\\Dataset\\mlp\\iris_train.txt", "..\\Dataset\\mlp\\iris_test.txt");

	return 0;
}

// The K-Nearest Neighborhood Classifier
void test_knn(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, "\t", LABEL_REAR);
	mlldata testset(testPath, "\t", LABEL_REAR);

	// Set a train condition
	const int K = 10;
	const measure& meas = correlation();

	// Train the dataset
	KNN knn(K, meas);
	knn.train(trainset);
	knn.save("..\\knn_result.ini");
// 	KNN knn;
// 	knn.open("..\\knn_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = knn.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = knn.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Naive Bayesian Classifier
void test_naivebayes(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, ",", LABEL_REAR);
	mlldata testset(testPath, ",", LABEL_REAR);

	// Train the dataset
	naivebayes nb;
	nb.train(trainset);
	nb.save("..\\naive_bayesian_result.ini");
// 	naivebayes nb;
// 	nb.open("..\\naive_bayesian_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = nb.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = nb.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Normal Bayesian Classifier
void test_normalbayes(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, ",", LABEL_REAR);
	mlldata testset(testPath, ",", LABEL_REAR);

	// Train the dataset
	normalbayes nb;
	nb.train(trainset);
	nb.save("..\\normal_bayesian_result.ini");
// 	normalbayes nb;
// 	nb.open("..\\normal_bayesian_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = nb.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = nb.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Support Vector Machine Classifier
void test_svm(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, "\t", LABEL_REAR);
	mlldata testset(testPath, "\t", LABEL_REAR);

	// Set a train condition
	const double C = 50.0;
	const double toler = 0.0001;
	const int maxIter = 30000;
// 	kernel& kn = linear_kernel();
// 	kernel& kn = polynomial_kernel(2.0, 1.0);
// 	kernel& kn = tanh_kernel(0.1, -0.1);
	kernel& kn = rbf_kernel(1.0);

	// Train the dataset
	SVM svm(C, toler, maxIter, kn);
	svm.train(trainset);
	svm.save("..\\svm_result.ini");
// 	SVM svm;
// 	svm.open("..\\svm_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = svm.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = svm.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Adaptive Boosting Classifier
void test_adaboost(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, "\t", LABEL_REAR);
	mlldata testset(testPath, "\t", LABEL_REAR);

	// Set a train condition
	const int nwc = 1000;

	// Train the dataset
	adaboost abst(nwc);
	abst.train(trainset);
	abst.save("..\\abst_result.ini");
// 	adaboost abst;
//	abst.open("..\\abst_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = abst.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = abst.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Logit Model (Logistic Regression) Classifier
void test_logitmodel(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, "\t", LABEL_REAR);
	mlldata testset(testPath, "\t", LABEL_REAR);

	// Set a train condition
	const int maxIter = 100;
	const double E = 0.01;

	// Train the dataset
	logitmodel lm(maxIter, E);
	lm.train(trainset);
	lm.save("..\\logit_model_result.ini");
// 	logitmodel lm;
// 	lm.open("..\\logit_model_result.ini");

	// Get a response on the train dataset
	int trainMissed = 0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		double response = lm.predict(trainset[0].row(i));
		if (response != trainset[1][i])
		{
			trainMissed++;
		}
	}

	// Get a response on the test dataset
	int testMissed = 0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		double response = lm.predict(testset[0].row(i));
		if (response != testset[1][i])
		{
			testMissed++;
		}
	}

	// Show the classification results
	cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
	cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
}

// The Multi Layer Perceptron Classifier
void test_mlp(string trainPath, string testPath)
{
	// Set a MLP mode
	const int mode = MLP_TRAIN_CLASSIFICATION;

	// Load the dataset
	mlldata trainset(trainPath, ",", LABEL_REAR);
	mlldata testset(testPath, ",", LABEL_REAR);
	trainset.shuffle();
	testset.shuffle();

	// Check the MLP mode
	if (mode == MLP_TRAIN_CLASSIFICATION)
	{
		// Set a train condition
		const int N = 10;
		const double E = 0.0001;
		const int maxIter = 5000;
		vector<netlayer> hl;
		hl.push_back(netlayer(10, nn::sigmoid()));
		const double mu = 0.0;
		const double sigma = 0.01;
		initializer* init = new initializer(mu, sigma, GAUSSIAN_WEIGHT_MANUAL);
		const double epsilon = 0.01;
		const double delta = 0.9;
		const double decay = 0.95;
		const double beta1 = 0.9;
		const double beta2 = 0.999;
		const int cycle = 10;
		const double k = 0.99;
 		optimizer* opt = new sgd(epsilon);
// 		optimizer* opt = new momentum(epsilon, delta);
// 		optimizer* opt = new nesterov(epsilon, delta);
// 		optimizer* opt = new adagrad(epsilon);
// 		optimizer* opt = new rmsprop(epsilon, decay);
// 		optimizer* opt = new adadelta(decay);
//		optimizer* opt = new adam(epsilon, beta1, beta2);
//		opt->set(cycle, k, ANNEALING_STEP);
		const double lamda = 0.0;
		vector<dropout> dolayers;
		dolayers.push_back(dropout(1.0));
		vector<batchnorm> bnlayers;
		bnlayers.push_back(batchnorm(true));
		regularizer* reg = new regularizer((int)hl.size(), lamda, REGULARIZE_NONE, dolayers, bnlayers);

		// Train the dataset
		MLP mlp(mode, N, E, maxIter, hl, *init, *opt, *reg);
		mlp.trD = trainset;
		mlp.teD = testset;
		mlp.progInterval = 10;
		mlp.train(trainset);
		mlp.save("..\\mlp_result.ini");
// 		MLP mlp;
// 		mlp.open("..\\mlp_result.ini");

		// Save a learning curve
		FILE* writer;
		fopen_s(&writer, "..\\learning_curve.csv", "w");
		for (int i = 0; i < (int)mlp.trC.size(); i++)
		{
			fprintf_s(writer, "%d,%lf,%lf\n", i, mlp.trC[i], mlp.teC[i]);
		}
		fclose(writer);

		// Get a response on the train dataset
		int trainMissed = 0;
		for (int i = 0; i < trainset[0].rows; i++)
		{
			double response = mlp.predict(trainset[0].row(i));
			if (response != trainset[1][i])
			{
				trainMissed++;
			}
		}

		// Get a response on the test dataset
		int testMissed = 0;
		for (int i = 0; i < testset[0].rows; i++)
		{
			double response = mlp.predict(testset[0].row(i));
			if (response != testset[1][i])
			{
				testMissed++;
			}
		}

		// Release the initializer and the optimizer
		delete init;
		delete opt;
		delete reg;

		// Show the classification results
		cout << "Error Rate on Trainset : " << (double)trainMissed / trainset[0].rows << endl;
		cout << "Error Rate on Testset : " << (double)testMissed / testset[0].rows << endl;
	}
	else
	{
		// Set test polynomial paramters
		const int numCoeffs = 2;
		const double c[numCoeffs] = { 121.7, 3.14 };

		// Create random generators
		random_device dev;
		mt19937 gen(dev());
		uniform_real_distribution<double> rand;
		normal_distribution<double> randn;

		// Generate the train dataset
		numat train_x(msize(trainset[0].rows));
		numat train_t(msize(trainset[0].rows));
		for (int i = 0; i < trainset[0].rows; i++)
		{
			train_x(i) = rand(gen);
			train_t(i) = 0.0;
			for (int j = numCoeffs - 1, k = 0; j >= 0; j--, k++)
			{
				train_t(i) += pow(train_x(i), j) * c[k];
			}
		}
		trainset.set(train_x, train_t);

		// Generate the test dataset
		numat test_x(msize(testset[0].rows));
		numat test_t(msize(testset[0].rows));
		for (int i = 0; i < testset[0].rows; i++)
		{
			test_x(i) = rand(gen);
			test_t(i) = 0.0;
			for (int j = numCoeffs - 1, k = 0; j >= 0; j--, k++)
			{
				test_t(i) += pow(test_x(i), j) * c[k];
			}
		}
		testset.set(test_x, test_t);

		// Set a train condition
		const int N = 10;
		const double E = 0.001;
		const int maxIter = 1000;
		vector<netlayer> hl;
//		hl.push_back(netlayer(5, nn::tanh()));
		const double mu = 0.0;
		const double sigma = 1.0;
		initializer* init = new initializer(mu, sigma, GAUSSIAN_WEIGHT_AUTO);
		const double epsilon = 0.01;
		const double delta = 0.9;
		const double decay = 0.99;
		const double beta1 = 0.9;
		const double beta2 = 0.999;
		optimizer* opt = new sgd(epsilon);
//		optimizer* opt = new momentum(epsilon, delta);
// 		optimizer* opt = new nag();
// 		optimizer* opt = new adagrad();
// 		optimizer* opt = new rmsprop();
// 		optimizer* opt = new adadelta();
// 		optimizer* opt = new adam();
		const int cycle = 10;
		const double k = 0.99;
		opt->set(cycle, k, ANNEALING_UNKNOWN);
		const double lamda = 0.0;
		vector<dropout> dolayers;
//		dolayers.push_back(dropout(1.0));
		vector<batchnorm> bnlayers;
//		bnlayers.push_back(batchnorm(false));
		regularizer* reg = new regularizer((int)hl.size(), lamda, REGULARIZE_NONE, dolayers, bnlayers);

		// Train the dataset
		MLP mlp(mode, N, E, maxIter, hl, *init, *opt, *reg);
		mlp.trD = trainset;
		mlp.teD = testset;
		mlp.progInterval = 10;
		mlp.train(trainset);
		mlp.save("..\\mlp_result.ini");
// 		MLP mlp;
// 		mlp.open("..\\mlp_result.ini");

		// Save a learning curve
		FILE* writer;
		fopen_s(&writer, "..\\learning_curve.csv", "w");
		for (int i = 0; i < (int)mlp.trC.size(); i++)
		{
			fprintf_s(writer, "%d,%lf\n", i, mlp.trC[i]);
		}
		fclose(writer);

		// Get a response on the train dataset
		double trainDelta = 0;
		for (int i = 0; i < trainset[0].rows; i++)
		{
			double response = mlp.predict(trainset[0].row(i));
			trainDelta += sqrt((response - trainset[1][i]) * (response - trainset[1][i]));
		}

		// Get a response on the test dataset
		double testDelta = 0;
		for (int i = 0; i < testset[0].rows; i++)
		{
			double response = mlp.predict(testset[0].row(i));
			testDelta += sqrt((response - testset[1][i]) * (response - testset[1][i]));
		}

		// Release the initializer and the optimizer
		delete init;
		delete opt;
		delete reg;

		// Show the regression results
		cout << "Error Rate on Trainset : " << (double)trainDelta / trainset[0].rows << endl;
		cout << "Error Rate on Testset : " << (double)testDelta / testset[0].rows << endl;
	}
}

// The Auto Encoder Generation Network
void test_ae(string trainPath, string testPath)
{
	// Load the dataset
	mlldata trainset(trainPath, ",", LABEL_EMPTY);
	mlldata testset(testPath, ",", LABEL_EMPTY);
	trainset.shuffle();
	testset.shuffle();

	// Set a train condition
	const int N = 10;
	const double E = 0.0001;
	const int maxIter = 1000;
	vector<netlayer> hl;
	hl.push_back(netlayer(10, nn::sigmoid()));
	const double mu = 0.0;
	const double sigma = 0.01;
	initializer* init = new initializer(mu, sigma, GAUSSIAN_WEIGHT_AUTO);
	const double epsilon = 0.001;
	const double delta = 0.9;
	const double decay = 0.95;
	const double beta1 = 0.9;
	const double beta2 = 0.999;
	const int cycle = 10;
	const double k = 0.99;
// 	optimizer* opt = new sgd(epsilon);
// 	optimizer* opt = new momentum(epsilon, delta);
// 	optimizer* opt = new nesterov(epsilon, delta);
// 	optimizer* opt = new adagrad(epsilon);
// 	optimizer* opt = new rmsprop(epsilon, decay);
// 	optimizer* opt = new adadelta(decay);
	optimizer* opt = new adam(epsilon, beta1, beta2);
	opt->set(cycle, k, ANNEALING_STEP);
	const double lamda = 0.0;
	const double beta = 1.0;
	const double rho = 0.05;
	const double momentum = 0.9;
	vector<dropout> dolayers;
// 	dolayers.push_back(dropout(1.0, DROPOUT_INPUT_LAYER));
	vector<batchnorm> bnlayers;
// 	bnlayers.push_back(batchnorm(false));
	regularizer* reg = new regularizer((int)hl.size(), lamda, beta, rho, momentum, REGULARIZE_NONE, dolayers, bnlayers);

	// Train the dataset
	autoEncoder ae(N, E, maxIter, hl, *init, *opt, *reg);
	ae.trD = trainset;
	ae.teD = testset;
	ae.progInterval = 10;
	ae.train(trainset);
	ae.save("..\\ae_result.ini");
// 	autoEncoder ae;
//	ae.open("..\\ae_result.ini");

	// Save a learning curve
	FILE* writer;
	fopen_s(&writer, "..\\learning_curve.csv", "w");
	for (int i = 0; i < (int)ae.trC.size(); i++)
	{
		fprintf_s(writer, "%d,%lf,%lf\n", i, ae.trC[i], ae.teC[i]);
	}
	fclose(writer);

	// Get a response on the train dataset
	double trainMSE = 0.0;
	for (int i = 0; i < trainset[0].rows; i++)
	{
		numat reconstruction = ae.generate(trainset[0].row(i));
		trainMSE += sqrt(numat::sum((reconstruction - trainset[1].row(i)).mul(reconstruction - trainset[1].row(i))));
	}

	// Get a response on the test dataset
	double testMSE = 0.0;
	for (int i = 0; i < testset[0].rows; i++)
	{
		numat reconstruction = ae.generate(testset[0].row(i));
		testMSE += sqrt(numat::sum((reconstruction - testset[1].row(i)).mul(reconstruction - testset[1].row(i))));
	}

	// Release the initializer and the optimizer
	delete init;
	delete opt;
	delete reg;

	// Show the classification results
	cout << "Mean Squared Error on Trainset : " << (double)trainMSE / trainset[0].rows << endl;
	cout << "Mean Squared Error on Testset : " << (double)testMSE / testset[0].rows << endl;
}
