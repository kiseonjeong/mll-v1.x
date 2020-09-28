#ifndef REGULARIZER_H
#define REGULARIZER_H

namespace mll
{
	// The Regularization Type
	typedef enum _rgtype
	{
		REGULARIZE_NONE = -1,
		REGULARIZE_L1,
		REGULARIZE_L2,
		REGULARIZE_SPARSITY,
	} rgtype;

	// The Regularizer
	class regularizer : public nml::object
	{
		// Variables
	public:
		// Number of hidden layers
		nml::prop::get<int> nhids;
		// Weight decay parameter
		nml::prop::get<double> lamda;
		// Regularization type
		nml::prop::get<int> type;
		// Sparsity parameters
		nml::prop::get<double> beta;
		nml::prop::get<double> rho;
		nml::prop::get<double> momentum;
		// Dropout layer architecture
		std::vector<mll::dropout> dolayers;
		// Batchnorm layer architecture
		std::vector<mll::batchnorm> bnlayers;

		// Functions
	public:
		// Set the regularization parameters
		// nhids : number of hidden layers
		// lamda : L2 regularization parameter (0.01 ~ 0.00001)
		// type : regularization type
		// dolayers : dropout layer architecture
		// bnlayers : batch normalization layer architecture
		void set(const int nhids, const double lamda, const rgtype type, const std::vector<mll::dropout>& dolayers = std::vector<mll::dropout>(), const std::vector<mll::batchnorm>& bnlayers = std::vector<mll::batchnorm>());
		// Set the regularization parameters
		// nhids : number of hidden layers
		// lamda : L2 regularization parameter (0.01 ~ 0.00001)
		// beta : sparsity parameter
		// rho : activation probability
		// momentum : momentum for sparsity (0.9)
		// type : regularization type
		// dolayers : dropout layer architecture
		// bnlayers : batch normalization layer architecture
		void set(const int nhids, const double lamda, const double beta, const double rho, const double momentum, const rgtype type, const std::vector<mll::dropout>& dolayers = std::vector<mll::dropout>(), const std::vector<mll::batchnorm>& bnlayers = std::vector<mll::batchnorm>());

		// Operators
	public:
		regularizer& operator=(const regularizer& obj);

		// Constructors & Destructor
	public:
		regularizer();
		regularizer(const int nhids, const double lamda, const rgtype type, const std::vector<mll::dropout>& dolayers = std::vector<mll::dropout>(), const std::vector<mll::batchnorm>& bnlayers = std::vector<mll::batchnorm>());
		regularizer(const int nhids, const double lamda, const double beta, const double rho, const double momentum, const rgtype type, const std::vector<mll::dropout>& dolayers = std::vector<mll::dropout>(), const std::vector<mll::batchnorm>& bnlayers = std::vector<mll::batchnorm>());
		regularizer(const regularizer& obj);
		~regularizer();

		// Variables
	private:
		// Number of hidden layers
		int _nhids;
		// Weight decay parameter
		double _lamda;
		// Regularization type
		int _type;
		// Sparsity parameters
		double _beta;
		double _rho;
		double _momentum;

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