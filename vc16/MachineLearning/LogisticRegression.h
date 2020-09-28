#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

namespace mll
{
	// The Logistic Regression Classifier
	class logitmodel : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// Set a train condition
		void condition(const int maxIter, const double E);
		// Train the dataset
		void train(const mlldata& dataset);
		void train(const mlldata& dataset, const int maxIter, const double E);
		// Predict a response
		const double predict(const nml::numat& x);
		const double predict(const nml::numat& x, double* score);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		logitmodel& operator=(const logitmodel& obj);

		// Constructors & Destructor
	public:
		logitmodel();
		logitmodel(const int maxIter, const double E);
		logitmodel(const mlldata& dataset, const int maxIter, const double E);
		logitmodel(const logitmodel& obj);
		~logitmodel();

		// Variables
	private:
		// Max iteration
		int maxIter;
		// Error rate
		double E;
		// Weight matrix
		nml::numat W;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Activation function
		const double sigmoid(const double x);
		// Open train information
		const int openTrainInfo(const std::string path, const std::string prefix);
		// Open weight information
		const int openWeightInfo(const std::string path, const std::string prefix);

	};
}

#endif