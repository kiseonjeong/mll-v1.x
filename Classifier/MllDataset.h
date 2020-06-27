#ifndef MLL_DATASET_H
#define MLL_DATASET_H

namespace mll
{
	// The Dataset for Machine Learning
	class mlldata : public mllutil
	{
		// Variables
	public:
		// Feature vector X
		nml::prop::get<nml::numat> X;
		// Target vector T
		nml::prop::get<nml::numat> T;
		// Dimension of feature vector
		nml::prop::get<int> dimension;
		// Number of sample
		nml::prop::get<int> nsample;
		// Class matrix
		nml::prop::get<nml::numat> C;
		// Number of class
		nml::prop::get<int> nclass;
		// Slope vector
		nml::prop::get<nml::numat> slope;
		// Intercept vector
		nml::prop::get<nml::numat> intercept;

		// Functions
	public:
		// Open the dataset
		const int open(const std::string path, const std::string separator = ",", const labelpos mode = LABEL_REAR);
		// Set the dataset
		void set(const nml::numat& X, const nml::numat& T);
		// Set the dataset
		void set(const nml::numat& X);
		// Check the dataset is empty or not
		const bool empty() const;
		// Shuffle the dataset
		void shuffle(const int iter = -1);
		// Scale the dataset
		void scale();
		// Normalize the dataset
		void normalize();
		// Generate the sub-datasets
		const std::vector<mlldata> subdata(const int subsize, const bool shuffling = true) const;

		// Operators
	public:
		mlldata& operator=(const mlldata& obj);
		const nml::numat operator[](const int idx) const;

		// Constructors & Destructor
	public:
		mlldata();
		mlldata(const std::string path, const std::string separator = ",", const labelpos mode = LABEL_REAR);
		mlldata(const nml::numat& X, const nml::numat& T);
		mlldata(const nml::numat& X);
		mlldata(const mlldata& obj);
		~mlldata();

		// Variables
	private:
		// Feature vector X
		nml::numat _X;
		// Target vector T
		nml::numat _T;
		// Dimension of feature vector
		int _dimension;
		// Number of sample
		int _nsample;
		// Class matrix
		nml::numat _C;
		// Number of class
		int _nclass;
		// Slope vector
		nml::numat _slope;
		// Intercept vector
		nml::numat _intercept;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();

	};
}

#endif