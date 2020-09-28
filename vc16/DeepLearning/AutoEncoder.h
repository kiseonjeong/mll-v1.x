#ifndef AUTO_ENCODER_H
#define AUTO_ENCODER_H

namespace mll
{
	// The Auto Encoder
	class autoEncoder : public MLP
	{
		// Variables
	public:
		

		// Functions
	public:
		// Set a train condition
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		void condition(const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// Train the dataset
		// dataset : training dataset
		// type : training type (0 = classification, 1 = regression)
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		void train(const mlldata& dataset);
		void train(const mlldata& dataset, const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// Generate a response
		const nml::numat generate(const nml::numat& x);
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		autoEncoder& operator=(const autoEncoder& obj);

		// Constructors & Destructor
	public:
		autoEncoder();
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		autoEncoder(const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// dataset : training dataset
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		autoEncoder(const mlldata& dataset, const int N, const double E, const int maxIter, const std::vector<netlayer>& hls, const initializer& init, const optimizer& opt, const regularizer& reg);
		autoEncoder(const autoEncoder& obj);
		~autoEncoder();

		// Variables
	private:
		// Sparsity parameters
		std::vector<nml::numat> rho0;
		std::vector<nml::numat> rho1;
		std::vector<nml::numat> KL;
		bool sparsity;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Backup the dataset
		void backupDataset(const mlldata& dataset);
		// Create a network architecture
		void createNetwork();
		// Initialize the sparsity parameters
		void initSparsity();
		// Compute forward propagations
		const double computeForwardProps(const mlldata& dataset, nml::numat& target, const bool tr = true);
		// Set an input and an output vector
		void setInoutVectors(const mlldata& dataset, nml::numat& target);
		// Calculate a cost value on the Mean Squared Error
		const double calculateMeanSquaredError(const nml::numat& y, const nml::numat& t, const bool tr = true) const;
		// Compute backward propagations
		void computeBackwardProps(const nml::numat& target);
		// Open train condition information
		const int openTrainCondInfo(const std::string path, const std::string prefix);
		// Open label information
		const int openLabelInfo(const std::string path, const std::string prefix);
		// Open layer information
		const int openLayerInfo(const std::string path, const std::string prefix);
		// Open the hidden architecture
		const int openHiddenArchitecture(const std::string path, const std::string prefix);
		// Open weight information
		const int openWeightInfo(const std::string path, const std::string prefix);

	};
}

#endif