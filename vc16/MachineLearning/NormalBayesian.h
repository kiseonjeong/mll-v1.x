#ifndef NORMAL_BAYESIAN_H
#define NORMAL_BAYESIAN_H

namespace mll
{
	// The Normal Bayesian Classifier
	class normalbayes : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// Train the dataset
		void train(const mlldata& dataset);
		// Predict a response
		const double predict(const nml::numat& x);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		normalbayes& operator=(const normalbayes& obj);

		// Constructors & Destructor
	public:
		normalbayes();
		normalbayes(const mlldata& dataset);
		normalbayes(const normalbayes& obj);
		~normalbayes();

		// Variables
	private:
		// Prior probability
		nml::numat prior;
		// Count matrix
		nml::numem<int> count;
		// Mean matrix
		nml::numem<nml::numat> mean;
		// Covariance matrix
		nml::numem<nml::numat> cov;
		// Inverse covariance matrix
		nml::numem<nml::numat> icov;
		// Class vector
		nml::numat C;
		// Rows length
		int vrows;
		// Column length
		int vcols;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Open label information
		const int openLabelInfo(const std::string path, const std::string prefix);
		// Open probability information
		const int openProbInfo(const std::string path, const std::string prefix);

	};
}

#endif