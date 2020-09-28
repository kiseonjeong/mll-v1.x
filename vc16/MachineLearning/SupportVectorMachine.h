#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

namespace mll
{
	// The Kernel Type
	typedef enum _kntype
	{
		SVM_KERNEL_UNKNOWN = -1,
		SVM_KERNEL_LINEAR,
		SVM_KERNEL_POLYNOMIAL,
		SVM_KERNEL_TANH,
		SVM_KERNEL_RBF,
	} kntype;

	// The Kernel for the Trick
	class kernel : public nml::object
	{
		// Variables
	public:
		// Kernel type
		nml::prop::get<kntype> type;
		// Degree for the polynomial kernel
		nml::prop::get<double> deg;
		// Constant Value for the polynomial kernel
		nml::prop::get<double> cons;
		// Threshold 1 for the hyperbolic tangent kernel
		nml::prop::get<double> th1;
		// Threshold 2 for the hyperbolic tangent kernel
		nml::prop::get<double> th2;
		// Sigma parameter for the RBF kernel
		nml::prop::get<double> sig;

		// Functions
	public:
		// Check the kernel is empty or not
		const bool empty() const;
		// Set a kernel for the trick
		virtual void set(const double deg, const double cons, const double th1, const double th2, const double sigma) = 0;
		// Remap the feature vectors using the kernel trick
		virtual const nml::numat trick(const nml::numat& X, const nml::numat& Xi) const = 0;

		// Constructors & Destructor
	public:
		kernel();
		virtual ~kernel();

		// Variables
	protected:
		// Kernel type
		kntype _type;
		// Degree for the polynomial kernel
		double _deg;
		// Constant Value for the polynomial kernel
		double _cons;
		// Threshold 1 for the hyperbolic tangent kernel
		double _th1;
		// Threshold 2 for the hyperbolic tangent kernel
		double _th2;
		// Sigma parameter for the RBF kernel
		double _sig;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();

	};

	// The Linear Kernel
	class linear_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// Remap the feature vectors using the kernel trick
		const nml::numat trick(const nml::numat& X, const nml::numat& Xi) const;

		// Operators
	public:
		linear_kernel& operator=(const linear_kernel& obj);

		// Constructors & Destructor
	public:
		linear_kernel();
		linear_kernel(const linear_kernel& obj);
		~linear_kernel();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Set a kernel for the trick
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);

	};

	// The Polynomial Kernel
	class polynomial_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// Set a kernel for the trick
		// cons > 0
		void set(const double deg, const double cons = 1.0);
		// Remap the feature vectors using the kernel trick
		const nml::numat trick(const nml::numat& X, const nml::numat& Xi) const;

		// Operators
	public:
		polynomial_kernel& operator=(const polynomial_kernel& obj);

		// Constructors & Destructor
	public:
		polynomial_kernel();
		// cons > 0
		polynomial_kernel(const double deg, const double cons);
		polynomial_kernel(const polynomial_kernel& obj);
		~polynomial_kernel();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Set a kernel for the trick
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);

	};

	// The Hyperbolic Tangent Kernel
	class tanh_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// Set a kernel for the trick
		// th1, th2 >= 0
		void set(const double th1, const double th2 = 0.0);
		// Remap the feature vectors using the kernel trick
		const nml::numat trick(const nml::numat& X, const nml::numat& Xi) const;

		// Operators
	public:
		tanh_kernel& operator=(const tanh_kernel& obj);

		// Constructors & Destructor
	public:
		tanh_kernel();
		// th1, th2 >= 0
		tanh_kernel(const double th1, const double th2 = 0.0);
		tanh_kernel(const tanh_kernel& obj);
		~tanh_kernel();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Set a kernel for the trick
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);

	};

	// The RBF Kernel
	class rbf_kernel : public kernel
	{
		// Variables
	public:


		// Functions
	public:
		// Set a kernel for the trick
		// Sigma != 0
		void set(const double sig);
		// Remap the feature vectors using the kernel trick
		const nml::numat trick(const nml::numat& X, const nml::numat& Xi) const;

		// Operators
	public:
		rbf_kernel& operator=(const rbf_kernel& obj);

		// Constructors & Destructor
	public:
		rbf_kernel();
		// Sigma != 0
		rbf_kernel(const double sig);
		rbf_kernel(const rbf_kernel& obj);
		~rbf_kernel();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Set a kernel for the trick
		void set(const double deg, const double cons, const double th1, const double th2, const double sig);

	};

	// The Support Vector Machine Classifier
	class SVM : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// Set a train condition
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		void condition(const double C, const double toler, const int maxIter, const kernel& kn);
		// Train the dataset
		// dataset : training dataset
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		void train(const mlldata& dataset);
		void train(const mlldata& dataset, const double C, const double toler, const int maxIter, const kernel& kn);
		// Predict a response
		const double predict(const nml::numat& x);
		const double predict(const nml::numat& x, double* distance);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		SVM& operator=(const SVM& obj);

		// Constructors & Destructor
	public:
		SVM();
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		SVM(const double C, const double toler, const int maxIter, const kernel& kn);
		// dataset : training dataset
		// C : SVM margin parameter (small = soft margin, big = hard margin)
		// toler : tolerance
		// maxIter : maximum iterations
		// kn : kernel for trick
		SVM(const mlldata& dataset, const double C, const double toler, const int maxIter, const kernel& kn);
		SVM(const SVM& obj);
		~SVM();

		// Variables
	private:
		// Panelty term for the soft margin
		double C;
		// Tolerence parameter
		double toler;
		// Maximum iterations
		int maxIter;
		// Kernel function
		kernel* kn;
		// Lagrange multipliers
		nml::numat A;
		// Normal Vector W
		nml::numat W;
		// Bias Paramter b
		double b;
		// Feature vector
		nml::numat X;
		// Target vector
		nml::numat T;
		// Error cache
		nml::numat E;
		// Kernel transform matrix
		nml::numat K;
		// The number of the support vectors
		int svC;
		// Feature vector X for the support vectors
		nml::numat svX;
		// Target vector T for the support vectors
		nml::numat svT;
		// Alpha parameters for the support vectors
		nml::numat svA;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Create a kernel function
		void createKernel(const int type, const double deg, const double cons, const double th1, const double th2, const double sig);
		// Copy the kernel function
		void copyKernel(const kernel& kn);
		// Remap the dataset using the kernel function
		void remapDataset(const nml::numat& X);
		// The Sequential Minimal Optimization Algorithm
		const int SMO(const mlldata& dataset);
		// Find a non-bound alpha list
		const std::vector<int> findNonBound();
		// Optimize an alpha pair
		const int optimizeAlphaPair(const int i);
		// Calculate error
		const double calculateError(const int idx);
		// Update error
		void updateError(const int idx);
		// Select another alpha
		void selectAnotherAlpha(const int i, const double Ei, int& j, double& Ej);
		// Get a random number
		const int getRandomNumber(const int i, const int min, const int max);
		// Clip the alpha value
		const double clipAlpha(const double alpha, const double H, const double L);
		// Calculate a normal vector
		const nml::numat calculateW();
		// Get the result parameters on the support vectors
		void getSupportVectorParams();
		// Open train condition information
		const int openTrainCondInfo(const std::string path, const std::string prefix);
		// Open kernel information
		const int openKernelInfo(const std::string path, const std::string prefix);
		// Open support vectors information
		const int openSupportVectorInfo(const std::string path, const std::string prefix);
		// Open label information
		const int openLabelInfo(const std::string path, const std::string prefix);
		// Open alpha information
		const int openAlphaInfo(const std::string path, const std::string prefix);

	};
}

#endif