#ifndef K_NEAREST_NEIGHBOR_H
#define K_NEAREST_NEIGHBOR_H

namespace mll
{
	// The Measurement Type
	typedef enum _meastype
	{
		KNN_MEASURE_UNKNOWN = -1,
		KNN_MEASURE_EUCLIDEAN,
		KNN_MEASURE_MANHATTAN,
		KNN_MEASURE_CHEBYCHEV,
		KNN_MEASURE_MINKOWSKI,
		KNN_MEASURE_COSINE,
		KNN_MEASURE_CORRELATION,
	} meastype;

	// The Measurement for a Distance
	class measure : public nml::object
	{
		// Variables
	public:
		// Measure type
		nml::prop::get<meastype> type;
		// Normalization flag
		nml::prop::get<bool> norm;
		// Normalization gain
		nml::prop::get<double> gain;
		// Normalization offset
		nml::prop::get<double> offset;
		// Power parameter for a distance
		nml::prop::get<double> p;

		// Functions
	public:
		// Calculate a distance
		virtual const double calculate(const nml::numat& xi, const nml::numat& xj) const = 0;
		// Set a measurement
		virtual void set(const bool norm, const double gain, const double offset, const double p) = 0;

		// Constructors & Destructor
	public:
		measure();
		virtual ~measure();

		// Variables
	protected:
		// Measure type
		meastype _type;
		// Normalization flag
		bool _norm;
		// Normalization gain
		double _gain;
		// Normalization offset
		double _offset;
		// Power parameter for a distance
		double _p;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();

	};

	// The Euclidean Measurement
	class euclidean : public measure
	{
		// Variables
	public:


		// Functions
	public:
		// Set a measurement
		void set(const bool norm, const double gain, const double offset);
		// Calculate a distance
		virtual const double calculate(const nml::numat& xi, const nml::numat& xj) const;

		// Operators
	public:
		euclidean& operator=(const euclidean& obj);

		// Constructors & Destructor
	public:
		euclidean();
		euclidean(const bool norm, const double gain, const double offset);
		euclidean(const euclidean& obj);
		virtual ~euclidean();

		// Variables
	protected:


		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Set a measurement
		virtual void set(const bool norm, const double gain, const double offset, const double p);

	};

	// The Manhattan Measurement
	class manhattan : public euclidean
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		manhattan& operator=(const manhattan& obj);

		// Constructors & Destructor
	public:
		manhattan();
		manhattan(const bool norm, const double gain, const double offset);
		manhattan(const manhattan& obj);
		~manhattan();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Minkowski Measurement
	class minkowski : public euclidean
	{
		// Variables
	public:


		// Functions
	public:
		// Set a measurement
		void set(const bool norm, const double gain, const double offset, const double p);

		// Operators
	public:
		minkowski& operator=(const minkowski& obj);

		// Constructors & Destructor
	public:
		minkowski();
		minkowski(const bool norm, const double gain, const double offset, const double p);
		minkowski(const minkowski& obj);
		~minkowski();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Chebychev Measurement
	class chebychev : public euclidean
	{
		// Variables
	public:


		// Functions
	public:
		// Calculate a distance
		const double calculate(const nml::numat& xi, const nml::numat& xj) const;

		// Operators
	public:
		chebychev& operator=(const chebychev& obj);

		// Constructors & Destructor
	public:
		chebychev();
		chebychev(const bool norm, const double gain, const double offset);
		chebychev(const chebychev& obj);
		~chebychev();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Cosine Measurement
	class cosine : public measure
	{
		// Variables
	public:


		// Functions
	public:
		// Calculate a distance
		const double calculate(const nml::numat& xi, const nml::numat& xj) const;

		// Operators
	public:
		cosine& operator=(const cosine& obj);

		// Constructors & Destructor
	public:
		cosine();
		cosine(const cosine& obj);
		virtual ~cosine();

		// Variables
	protected:


		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Set a measurement
		void set(const bool norm, const double gain, const double offset, const double p);

	};

	// The Correlation Measurement
	class correlation : public cosine
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		correlation& operator=(const correlation& obj);

		// Constructors & Destructor
	public:
		correlation();
		correlation(const correlation& obj);
		~correlation();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The K-Nearest Neighborhood Classifier
	class KNN : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// Set a train condition
		void condition(const int K, const measure& meas);
		// Train the dataset
		void train(const mlldata& dataset);
		void train(const mlldata& dataset, const int K, const measure& meas);
		// Predict a response
		const double predict(const nml::numat& x);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		KNN& operator=(const KNN& obj);

		// Constructors & Destructor
	public:
		KNN();
		KNN(const int K, const measure& meas);
		KNN(const mlldata& dataset, const int K, const measure& meas);
		KNN(const KNN& obj);
		~KNN();

		// Variables
	private:
		// Number of neighbors
		int K;
		// Distance measurement
		measure* meas;
		// Min vector
		nml::numat minVec;
		// Max vector
		nml::numat maxVec;
		// Mean vector
		nml::numat M;
		// Feature vector
		nml::numat X;
		// Target vector
		nml::numat T;
		// Class vector
		nml::numat C;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Create a measurement for a distance
		void createMeasure(const int type, const bool norm, const double gain, const double offset, const double p);
		// Copy the measurement for a distance
		void copyMeasure(const measure& meas);
		// Convert scale on the dataset
		void convertScale();
		void convertScale(nml::numat& x) const;
		// Open measurement information
		const int openMeasureInfo(const std::string path, const std::string prefix);
		// Open label information
		const int openLabelInfo(const std::string path, const std::string prefix);
		// Open sample information
		const int openSampleInfo(const std::string path, const std::string prefix);

	};
}

#endif