#ifndef BATCHNORM_H
#define BATCHNORM_H

namespace mll
{
	// The Batch Normalization Regularizer
	class batchnorm : public nml::object
	{
		// Variables
	public:
		// mu'
		nml::numat mu_p;
		// var'
		nml::numat var_p;
		// gamma
		nml::numat gamma;
		// beta
		nml::numat beta;
		// Activation flag
		nml::prop::get<bool> act;
		// Momentum parameter
		nml::prop::get<double> momentum;

		// Functions
	public:
		// Set parameters
		// act : activation flag
		// momentum : moving average parameter
		void set(const bool act, const double momentum = 0.9);
		// Activative the data
		const nml::numat forward(const nml::numat& net);
		// Derivative the data
		const nml::numat backward(const nml::numat& dout);
		// Update the learning parameters
		void update(const double epsilon);
		// Inference the data
		const nml::numat inference(const nml::numat& net);

		// Operators
	public:
		batchnorm& operator=(const batchnorm& obj);

		// Constructors & Destructor
	public:
		batchnorm();
		batchnorm(const bool act, const double momentum = 0.9);
		batchnorm(const batchnorm& obj);
		~batchnorm();

		// Variables
	private:
		// Activation flag
		bool _act;
		// Momentum parameter
		double _momentum;
		// Backpropagation parameters
		nml::numat mu;
		nml::numat var;
		nml::numat std;
		nml::numat X;
		nml::numat X_norm;
		nml::numat X_mu;
		nml::numat dX_norm;
		nml::numat dmu;
		nml::numat dvar;
		nml::numat stdi;
		nml::numat dgamma;
		nml::numat dbeta;

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