#ifndef INITIALIZER_H
#define INITIALIZER_H

namespace mll
{
	// The Gaussian Weight Mode
	typedef enum _gwmode
	{
		GAUSSIAN_WEIGHT_UNKNOWN = -1,
		GAUSSIAN_WEIGHT_MANUAL,
		GAUSSIAN_WEIGHT_XAVIER,
		GAUSSIAN_WEIGHT_HE,
		GAUSSIAN_WEIGHT_AUTO,
	} gwmode;

	// The Initializer
	class initializer : public nml::object
	{
		// Variables
	public:
		// Gaussian mean value
		nml::prop::get<double> mu;
		// Gaussian std. dev. value
		nml::prop::get<double> sigma;
		// Gaussian mode
		nml::prop::get<gwmode> mode;

		// Functions
	public:
		// Set the gaussian parameters
		// mu : mean of normal distribution
		// sigma : standard deviation of normal distribution
		// mode : weight initialization mode
		void set(const double mu, const double sigma, const gwmode mode);
		// Generate a weight matrix
		void generate(const int aftype, nml::numat& W) const;

		// Operators
	public:
		initializer& operator=(const initializer& obj);

		// Constructors & Destructor
	public:
		initializer();
		// mu : mean of normal distribution
		// sigma : standard deviation of normal distribution
		// mode : weight initialization mode
		initializer(const double mu, const double sigma, const gwmode mode);
		initializer(const initializer& obj);
		~initializer();

		// Variables
	private:
		// Gaussian mean value
		double _mu;
		// Gaussian std. dev. value
		double _sigma;
		// Gaussian mode
		gwmode _mode;

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